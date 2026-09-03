// C++ acceleration for residual-density isosurface extraction.
// This module is optional; callers should guard imports with try/except in the
// same style as fastmolwidget.sdm (HAS_CPP = False when the extension is absent).
//
// Build:
//   macOS  : uv pip install pybind11 && uv pip install -e . --no-build-isolation
//   Linux  : uv pip install pybind11 && uv pip install -e . --no-build-isolation
//   Windows: uv pip install pybind11 && uv pip install -e . --no-build-isolation
//
// Interface:
//
//   marching_cubes(
//       grid   : numpy.ndarray,               # shape (nx, ny, nz), float32/float64
//       level  : float,                       # isosurface level
//       origin : tuple[float, float, float],  # Cartesian position of grid[0,0,0]
//       step   : tuple[float, float, float],  # Cartesian spacing along x, y, z
//   ) -> tuple[numpy.ndarray, numpy.ndarray]
//
// Returns
// -------
// (vertices, edges)
//   vertices : float64 array, shape (M, 3)
//              unique isosurface vertices in Cartesian coordinates
//   edges    : int64 array, shape (K, 2)
//              unique undirected wireframe edges referencing `vertices`
//
// Algorithm
// ---------
// Classic Lorensen–Cline marching cubes on a regular 3-D scalar grid. Shared
// vertices are keyed by the canonical identity of the underlying grid edge, so
// neighbouring cubes re-use exactly the same vertex indices. The 256 marching-
// cubes case tables are generated inline from the standard cube topology so the
// module stays completely self-contained.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace {

constexpr int kCornerCount = 8;
constexpr int kEdgeCount = 12;
constexpr int kFaceCount = 6;
constexpr int kTriTableWidth = 16;
constexpr double kEpsilon = 1.0e-12;

constexpr std::array<std::array<int, 3>, kCornerCount> kCornerOffsets{{
    {{0, 0, 0}}, {{1, 0, 0}}, {{1, 1, 0}}, {{0, 1, 0}},
    {{0, 0, 1}}, {{1, 0, 1}}, {{1, 1, 1}}, {{0, 1, 1}},
}};

constexpr std::array<std::array<int, 2>, kEdgeCount> kEdgeCorners{{
    {{0, 1}}, {{1, 2}}, {{2, 3}}, {{3, 0}},
    {{4, 5}}, {{5, 6}}, {{6, 7}}, {{7, 4}},
    {{0, 4}}, {{1, 5}}, {{2, 6}}, {{3, 7}},
}};

constexpr std::array<std::array<int, 4>, kFaceCount> kFaceCorners{{
    {{0, 1, 2, 3}},  // z = 0
    {{4, 5, 6, 7}},  // z = 1
    {{0, 1, 5, 4}},  // y = 0
    {{1, 2, 6, 5}},  // x = 1
    {{2, 3, 7, 6}},  // y = 1
    {{3, 0, 4, 7}},  // x = 0
}};

constexpr std::array<std::array<int, 4>, kFaceCount> kFaceEdges{{
    {{0, 1, 2, 3}},
    {{4, 5, 6, 7}},
    {{0, 9, 4, 8}},
    {{1, 10, 5, 9}},
    {{2, 11, 6, 10}},
    {{3, 8, 7, 11}},
}};

struct LookupTables {
    std::array<int, 256> edge_masks{};
    std::array<std::array<int, kTriTableWidth>, 256> tri_table{};
};

struct GridEdgeKey {
    int x;
    int y;
    int z;
    int axis;

    bool operator==(const GridEdgeKey& other) const noexcept {
        return x == other.x && y == other.y && z == other.z && axis == other.axis;
    }
};

struct GridEdgeKeyHash {
    std::size_t operator()(const GridEdgeKey& key) const noexcept {
        std::size_t seed = 0;
        seed ^= static_cast<std::size_t>(key.x) + 0x9e3779b9u + (seed << 6) + (seed >> 2);
        seed ^= static_cast<std::size_t>(key.y) + 0x9e3779b9u + (seed << 6) + (seed >> 2);
        seed ^= static_cast<std::size_t>(key.z) + 0x9e3779b9u + (seed << 6) + (seed >> 2);
        seed ^= static_cast<std::size_t>(key.axis) + 0x9e3779b9u + (seed << 6) + (seed >> 2);
        return seed;
    }
};

struct WireEdge {
    std::int64_t a;
    std::int64_t b;

    WireEdge(std::int64_t first, std::int64_t second) noexcept {
        if (first <= second) {
            a = first;
            b = second;
        } else {
            a = second;
            b = first;
        }
    }

    bool operator==(const WireEdge& other) const noexcept {
        return a == other.a && b == other.b;
    }
};

struct WireEdgeHash {
    std::size_t operator()(const WireEdge& edge) const noexcept {
        std::size_t seed = 0;
        seed ^= static_cast<std::size_t>(edge.a) + 0x9e3779b9u + (seed << 6) + (seed >> 2);
        seed ^= static_cast<std::size_t>(edge.b) + 0x9e3779b9u + (seed << 6) + (seed >> 2);
        return seed;
    }
};

void add_connection(
    std::array<std::array<bool, kEdgeCount>, kEdgeCount>& adjacency,
    int edge_a,
    int edge_b
) {
    if (edge_a == edge_b) {
        return;
    }
    adjacency[edge_a][edge_b] = true;
    adjacency[edge_b][edge_a] = true;
}

LookupTables build_lookup_tables() {
    LookupTables tables;

    for (auto& row : tables.tri_table) {
        row.fill(-1);
    }

    for (int case_index = 0; case_index < 256; ++case_index) {
        int edge_mask = 0;
        for (int edge = 0; edge < kEdgeCount; ++edge) {
            const int corner_a = kEdgeCorners[edge][0];
            const int corner_b = kEdgeCorners[edge][1];
            const bool inside_a = ((case_index >> corner_a) & 1) != 0;
            const bool inside_b = ((case_index >> corner_b) & 1) != 0;
            if (inside_a != inside_b) {
                edge_mask |= (1 << edge);
            }
        }
        tables.edge_masks[case_index] = edge_mask;

        if (edge_mask == 0) {
            continue;
        }

        std::array<std::array<bool, kEdgeCount>, kEdgeCount> adjacency{};

        for (int face = 0; face < kFaceCount; ++face) {
            std::array<bool, 4> face_inside{};
            std::array<bool, 4> face_cut{};
            int cut_count = 0;

            for (int i = 0; i < 4; ++i) {
                const int corner = kFaceCorners[face][i];
                const int next_corner = kFaceCorners[face][(i + 1) % 4];
                face_inside[i] = ((case_index >> corner) & 1) != 0;
                face_cut[i] = (((case_index >> corner) & 1) != ((case_index >> next_corner) & 1));
                if (face_cut[i]) {
                    ++cut_count;
                }
            }

            if (cut_count == 2) {
                int first = -1;
                int second = -1;
                for (int i = 0; i < 4; ++i) {
                    if (!face_cut[i]) {
                        continue;
                    }
                    if (first < 0) {
                        first = i;
                    } else {
                        second = i;
                    }
                }
                add_connection(adjacency, kFaceEdges[face][first], kFaceEdges[face][second]);
            } else if (cut_count == 4) {
                if (face_inside[0]) {
                    add_connection(adjacency, kFaceEdges[face][3], kFaceEdges[face][0]);
                    add_connection(adjacency, kFaceEdges[face][1], kFaceEdges[face][2]);
                } else {
                    add_connection(adjacency, kFaceEdges[face][0], kFaceEdges[face][1]);
                    add_connection(adjacency, kFaceEdges[face][2], kFaceEdges[face][3]);
                }
            }
        }

        std::array<std::array<bool, kEdgeCount>, kEdgeCount> used{};
        int tri_pos = 0;

        for (int start = 0; start < kEdgeCount; ++start) {
            for (int neighbour = 0; neighbour < kEdgeCount; ++neighbour) {
                if (!adjacency[start][neighbour] || used[start][neighbour]) {
                    continue;
                }

                std::vector<int> loop;
                loop.reserve(6);
                int prev = -1;
                int curr = start;
                int next = neighbour;

                loop.push_back(curr);

                while (true) {
                    used[curr][next] = true;
                    used[next][curr] = true;
                    prev = curr;
                    curr = next;

                    if (curr == start) {
                        break;
                    }

                    loop.push_back(curr);

                    int candidate = -1;
                    for (int candidate_edge = 0; candidate_edge < kEdgeCount; ++candidate_edge) {
                        if (adjacency[curr][candidate_edge] && candidate_edge != prev && !used[curr][candidate_edge]) {
                            candidate = candidate_edge;
                            break;
                        }
                    }
                    if (candidate < 0) {
                        for (int candidate_edge = 0; candidate_edge < kEdgeCount; ++candidate_edge) {
                            if (adjacency[curr][candidate_edge] && candidate_edge != prev) {
                                candidate = candidate_edge;
                                break;
                            }
                        }
                    }
                    if (candidate < 0) {
                        loop.clear();
                        break;
                    }
                    next = candidate;
                }

                if (loop.size() < 3) {
                    continue;
                }

                for (std::size_t i = 1; i + 1 < loop.size() && tri_pos + 2 < kTriTableWidth; ++i) {
                    tables.tri_table[case_index][tri_pos++] = loop[0];
                    tables.tri_table[case_index][tri_pos++] = loop[i];
                    tables.tri_table[case_index][tri_pos++] = loop[i + 1];
                }
            }
        }
    }

    return tables;
}

const LookupTables& lookup_tables() {
    static const LookupTables tables = build_lookup_tables();
    return tables;
}

GridEdgeKey canonical_edge_key(int cube_x, int cube_y, int cube_z, int edge_id) {
    switch (edge_id) {
        case 0:  return {cube_x,     cube_y,     cube_z,     0};
        case 1:  return {cube_x + 1, cube_y,     cube_z,     1};
        case 2:  return {cube_x,     cube_y + 1, cube_z,     0};
        case 3:  return {cube_x,     cube_y,     cube_z,     1};
        case 4:  return {cube_x,     cube_y,     cube_z + 1, 0};
        case 5:  return {cube_x + 1, cube_y,     cube_z + 1, 1};
        case 6:  return {cube_x,     cube_y + 1, cube_z + 1, 0};
        case 7:  return {cube_x,     cube_y,     cube_z + 1, 1};
        case 8:  return {cube_x,     cube_y,     cube_z,     2};
        case 9:  return {cube_x + 1, cube_y,     cube_z,     2};
        case 10: return {cube_x + 1, cube_y + 1, cube_z,     2};
        case 11: return {cube_x,     cube_y + 1, cube_z,     2};
        default: throw std::logic_error("invalid marching-cubes edge id");
    }
}

template <typename T>
double sample_grid(const T* data, py::ssize_t ny, py::ssize_t nz, int x, int y, int z) {
    const auto offset = (static_cast<py::ssize_t>(x) * ny + static_cast<py::ssize_t>(y)) * nz + static_cast<py::ssize_t>(z);
    return static_cast<double>(data[offset]);
}

double interpolate_mu(double level, double value_a, double value_b) {
    if (std::abs(level - value_a) < kEpsilon) {
        return 0.0;
    }
    if (std::abs(level - value_b) < kEpsilon) {
        return 1.0;
    }
    if (std::abs(value_a - value_b) < kEpsilon) {
        return 0.0;
    }
    const double mu = (level - value_a) / (value_b - value_a);
    return std::clamp(mu, 0.0, 1.0);
}

// A coarse occupancy mask lets the caller confine the scan to the parts of the
// grid it actually wants contoured.  The box a molecule needs is its bounding
// box, which for anything but a compact blob is several times the molecule's
// own volume, so most of the grid holds density that is thrown away again
// straight after (see _clip_to_atoms on the Python side).  Skipping those
// cubes in whole blocks means they are never even read.
//
// The mask has one entry per block of `block` cubes along each axis, i.e.
// shape ceil((n - 1) / block).  Cube (x, y, z) is visited only when
// mask[x / block, y / block, z / block] is true.  Iteration order is
// unchanged, so the vertices are numbered exactly as they would be by an
// unmasked scan restricted to the same cubes.
struct CubeMask {
    const bool* data = nullptr;
    py::ssize_t shape[3] = {0, 0, 0};
    int block = 1;

    bool empty() const { return data == nullptr; }
};

py::tuple empty_result() {
    // The shape must be spelled out as a ShapeContainer: a braced list of two
    // ssize_t is ambiguous between array_t(ShapeContainer) and
    // array_t(const buffer_info&) in newer pybind11 releases.
    auto vertices = py::array_t<double>(py::array::ShapeContainer{0, 3});
    auto edges = py::array_t<std::int64_t>(py::array::ShapeContainer{0, 2});
    return py::make_tuple(vertices, edges);
}

template <typename T>
py::tuple marching_cubes_impl(
    const py::array_t<T, py::array::c_style>& grid,
    double level,
    const std::array<double, 3>& origin,
    const std::array<double, 3>& step,
    const CubeMask& mask
) {
    const py::buffer_info info = grid.request();
    const auto* data = static_cast<const T*>(info.ptr);
    const py::ssize_t nx = info.shape[0];
    const py::ssize_t ny = info.shape[1];
    const py::ssize_t nz = info.shape[2];

    if (nx < 2 || ny < 2 || nz < 2) {
        return empty_result();
    }

    const auto& tables = lookup_tables();
    std::unordered_map<GridEdgeKey, std::int64_t, GridEdgeKeyHash> vertex_lookup;
    std::vector<std::array<double, 3>> vertices;
    std::unordered_set<WireEdge, WireEdgeHash> edge_lookup;
    std::vector<WireEdge> edges;

    auto add_wire_edge = [&](std::int64_t first, std::int64_t second) {
        if (first == second) {
            return;
        }
        WireEdge edge(first, second);
        if (edge_lookup.insert(edge).second) {
            edges.push_back(edge);
        }
    };

    auto get_or_create_vertex = [&](int cube_x, int cube_y, int cube_z, int edge_id) -> std::int64_t {
        const GridEdgeKey key = canonical_edge_key(cube_x, cube_y, cube_z, edge_id);
        const auto it = vertex_lookup.find(key);
        if (it != vertex_lookup.end()) {
            return it->second;
        }

        const int corner_a = kEdgeCorners[edge_id][0];
        const int corner_b = kEdgeCorners[edge_id][1];

        const int x0 = cube_x + kCornerOffsets[corner_a][0];
        const int y0 = cube_y + kCornerOffsets[corner_a][1];
        const int z0 = cube_z + kCornerOffsets[corner_a][2];
        const int x1 = cube_x + kCornerOffsets[corner_b][0];
        const int y1 = cube_y + kCornerOffsets[corner_b][1];
        const int z1 = cube_z + kCornerOffsets[corner_b][2];

        const double value_a = sample_grid(data, ny, nz, x0, y0, z0);
        const double value_b = sample_grid(data, ny, nz, x1, y1, z1);
        const double mu = interpolate_mu(level, value_a, value_b);

        const double gx = static_cast<double>(x0) + mu * static_cast<double>(x1 - x0);
        const double gy = static_cast<double>(y0) + mu * static_cast<double>(y1 - y0);
        const double gz = static_cast<double>(z0) + mu * static_cast<double>(z1 - z0);

        const std::int64_t index = static_cast<std::int64_t>(vertices.size());
        vertices.push_back({
            origin[0] + gx * step[0],
            origin[1] + gy * step[1],
            origin[2] + gz * step[2],
        });
        vertex_lookup.emplace(key, index);
        return index;
    };

    const int cubes_x = static_cast<int>(nx - 1);
    const int cubes_y = static_cast<int>(ny - 1);
    const int cubes_z = static_cast<int>(nz - 1);
    const int block = mask.empty() ? 1 : mask.block;
    const py::ssize_t mask_stride_y = mask.empty() ? 0 : mask.shape[2];
    const py::ssize_t mask_stride_x = mask.empty() ? 0 : mask.shape[1] * mask.shape[2];

    for (int cube_x = 0; cube_x < cubes_x; ++cube_x) {
        for (int cube_y = 0; cube_y < cubes_y; ++cube_y) {
            const bool* mask_row = mask.empty() ? nullptr
                : mask.data
                    + static_cast<py::ssize_t>(cube_x / block) * mask_stride_x
                    + static_cast<py::ssize_t>(cube_y / block) * mask_stride_y;
            for (int cube_z = 0; cube_z < cubes_z; ++cube_z) {
                if (mask_row != nullptr && !mask_row[cube_z / block]) {
                    // Jump to the last cube of this block; the loop's own
                    // increment then lands on the first cube of the next one,
                    // so an unwanted block costs one test rather than one per
                    // cube - and its density is never read at all.
                    cube_z = (cube_z / block + 1) * block - 1;
                    continue;
                }

                int case_index = 0;

                for (int corner = 0; corner < kCornerCount; ++corner) {
                    const int gx = cube_x + kCornerOffsets[corner][0];
                    const int gy = cube_y + kCornerOffsets[corner][1];
                    const int gz = cube_z + kCornerOffsets[corner][2];
                    const double value = sample_grid(data, ny, nz, gx, gy, gz);
                    if (value < level) {
                        case_index |= (1 << corner);
                    }
                }

                const int edge_mask = tables.edge_masks[case_index];
                if (edge_mask == 0) {
                    continue;
                }

                std::array<std::int64_t, kEdgeCount> local_vertices{};
                local_vertices.fill(-1);
                for (int edge = 0; edge < kEdgeCount; ++edge) {
                    if ((edge_mask & (1 << edge)) == 0) {
                        continue;
                    }
                    local_vertices[edge] = get_or_create_vertex(cube_x, cube_y, cube_z, edge);
                }

                const auto& tri_row = tables.tri_table[case_index];
                for (int tri_pos = 0; tri_pos + 2 < kTriTableWidth && tri_row[tri_pos] != -1; tri_pos += 3) {
                    const std::int64_t v0 = local_vertices[tri_row[tri_pos]];
                    const std::int64_t v1 = local_vertices[tri_row[tri_pos + 1]];
                    const std::int64_t v2 = local_vertices[tri_row[tri_pos + 2]];
                    if (v0 < 0 || v1 < 0 || v2 < 0) {
                        continue;
                    }
                    if (v0 == v1 || v1 == v2 || v0 == v2) {
                        continue;
                    }
                    add_wire_edge(v0, v1);
                    add_wire_edge(v1, v2);
                    add_wire_edge(v2, v0);
                }
            }
        }
    }

    auto vertices_out = py::array_t<double>(py::array::ShapeContainer{
        static_cast<py::ssize_t>(vertices.size()), py::ssize_t(3)});
    auto edges_out = py::array_t<std::int64_t>(py::array::ShapeContainer{
        static_cast<py::ssize_t>(edges.size()), py::ssize_t(2)});

    auto vertices_view = vertices_out.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(vertices.size()); ++i) {
        vertices_view(i, 0) = vertices[static_cast<std::size_t>(i)][0];
        vertices_view(i, 1) = vertices[static_cast<std::size_t>(i)][1];
        vertices_view(i, 2) = vertices[static_cast<std::size_t>(i)][2];
    }

    auto edges_view = edges_out.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(edges.size()); ++i) {
        edges_view(i, 0) = edges[static_cast<std::size_t>(i)].a;
        edges_view(i, 1) = edges[static_cast<std::size_t>(i)].b;
    }

    return py::make_tuple(vertices_out, edges_out);
}

py::tuple marching_cubes(
    py::array grid,
    double level,
    std::array<double, 3> origin,
    std::array<double, 3> step,
    py::object mask_object,
    int block
) {
    if (grid.ndim() != 3) {
        throw py::value_error("grid must be a 3-D NumPy array");
    }
    if ((grid.flags() & py::array::c_style) == 0) {
        throw py::value_error("grid must be C-contiguous");
    }

    CubeMask mask;
    py::array_t<bool, py::array::c_style> mask_array;
    if (!mask_object.is_none()) {
        if (block < 1) {
            throw py::value_error("block must be >= 1");
        }
        mask_array = py::cast<py::array_t<bool, py::array::c_style>>(mask_object);
        if (mask_array.ndim() != 3) {
            throw py::value_error("mask must be a 3-D NumPy array");
        }
        for (int axis = 0; axis < 3; ++axis) {
            const py::ssize_t cubes = grid.shape(axis) - 1;
            const py::ssize_t wanted = cubes > 0 ? (cubes + block - 1) / block : 0;
            if (mask_array.shape(axis) != wanted) {
                throw py::value_error(
                    "mask shape must be ceil((grid shape - 1) / block) along every axis");
            }
        }
        mask.data = mask_array.data();
        mask.shape[0] = mask_array.shape(0);
        mask.shape[1] = mask_array.shape(1);
        mask.shape[2] = mask_array.shape(2);
        mask.block = block;
    }

    if (grid.dtype().is(py::dtype::of<float>())) {
        auto typed = py::reinterpret_borrow<py::array_t<float, py::array::c_style>>(grid);
        return marching_cubes_impl<float>(typed, level, origin, step, mask);
    }
    if (grid.dtype().is(py::dtype::of<double>())) {
        auto typed = py::reinterpret_borrow<py::array_t<double, py::array::c_style>>(grid);
        return marching_cubes_impl<double>(typed, level, origin, step, mask);
    }

    throw py::type_error("grid must have dtype float32 or float64");
}

// ---------------------------------------------------------------------------
// Direct summation of structure factors
// ---------------------------------------------------------------------------
//
// Same formula as gemmi's StructureFactorCalculator, but the phase factor
//     exp(2 pi i (h x + k y + l z))
// is not evaluated with a cos/sin pair per (reflection, symmetry image, site).
// Since h, k and l are integers, the factor separates into
//     Ex[h] * Ey[k] * Ez[l]
// and those three tables can be tabulated once per symmetry-transformed site
// over the Miller-index range of the data.  The inner loop is then two complex
// multiplications instead of two transcendental calls.

using Complex = std::complex<double>;

constexpr double kTwoPi = 6.283185307179586476925286766559;
constexpr double kTwoPiSq = 19.739208802178717237668981999752;  // 2 pi^2
constexpr double kUtoB = 78.956835208714868950675927999008;     // 8 pi^2

// Tabulate exp(2 pi i * n * value) for n in [low, high].
void fill_phase_table(double value, int low, int high, Complex* out) {
    for (int n = low; n <= high; ++n) {
        const double angle = kTwoPi * static_cast<double>(n) * value;
        out[n - low] = Complex(std::cos(angle), std::sin(angle));
    }
}

py::array_t<Complex> structure_factors(
    py::array_t<int, py::array::c_style | py::array::forcecast> hkl,
    py::array_t<double, py::array::c_style | py::array::forcecast> stol2,
    py::array_t<double, py::array::c_style | py::array::forcecast> rotations,
    py::array_t<double, py::array::c_style | py::array::forcecast> translations,
    py::array_t<double, py::array::c_style | py::array::forcecast> fract,
    py::array_t<double, py::array::c_style | py::array::forcecast> occupancies,
    py::array_t<double, py::array::c_style | py::array::forcecast> u_iso,
    py::array_t<double, py::array::c_style | py::array::forcecast> aniso,
    py::array_t<int, py::array::c_style | py::array::forcecast> form_index,
    py::array_t<double, py::array::c_style | py::array::forcecast> form_factors,
    std::array<double, 3> reciprocal
) {
    if (hkl.ndim() != 2 || hkl.shape(1) != 3) {
        throw py::value_error("hkl must have shape (N, 3)");
    }
    if (rotations.ndim() != 3 || rotations.shape(1) != 3 || rotations.shape(2) != 3) {
        throw py::value_error("rotations must have shape (M, 3, 3)");
    }
    if (fract.ndim() != 2 || fract.shape(1) != 3) {
        throw py::value_error("fract must have shape (S, 3)");
    }
    if (aniso.ndim() != 2 || aniso.shape(1) != 6) {
        throw py::value_error("aniso must have shape (S, 6)");
    }
    if (form_factors.ndim() != 2) {
        throw py::value_error("form_factors must have shape (F, N)");
    }

    const py::ssize_t n_refl = hkl.shape(0);
    const py::ssize_t n_ops = rotations.shape(0);
    const py::ssize_t n_sites = fract.shape(0);

    if (translations.ndim() != 2 || translations.shape(0) != n_ops
            || translations.shape(1) != 3) {
        throw py::value_error("translations must have shape (M, 3)");
    }
    if (stol2.shape(0) != n_refl || form_factors.shape(1) != n_refl) {
        throw py::value_error("stol2 and form_factors must cover every reflection");
    }
    if (occupancies.shape(0) != n_sites || u_iso.shape(0) != n_sites
            || aniso.shape(0) != n_sites || form_index.shape(0) != n_sites) {
        throw py::value_error("per-site arrays must cover every site");
    }

    auto result = py::array_t<Complex>(n_refl);
    if (n_refl == 0 || n_ops == 0 || n_sites == 0) {
        std::fill_n(result.mutable_data(), n_refl, Complex(0.0, 0.0));
        return result;
    }

    const int* hkl_data = hkl.data();
    const double* stol2_data = stol2.data();
    const double* rot_data = rotations.data();
    const double* tran_data = translations.data();
    const double* fract_data = fract.data();
    const double* occ_data = occupancies.data();
    const double* uiso_data = u_iso.data();
    const double* aniso_data = aniso.data();
    const int* form_index_data = form_index.data();
    const double* form_data = form_factors.data();

    int low[3] = {hkl_data[0], hkl_data[1], hkl_data[2]};
    int high[3] = {hkl_data[0], hkl_data[1], hkl_data[2]};
    for (py::ssize_t n = 1; n < n_refl; ++n) {
        for (int axis = 0; axis < 3; ++axis) {
            const int value = hkl_data[n * 3 + axis];
            low[axis] = std::min(low[axis], value);
            high[axis] = std::max(high[axis], value);
        }
    }
    const py::ssize_t span[3] = {
        high[0] - low[0] + 1, high[1] - low[1] + 1, high[2] - low[2] + 1,
    };

    // One symmetry image of one site, with its three phase tables.
    const py::ssize_t n_images = n_ops * n_sites;
    const py::ssize_t stride = span[0] + span[1] + span[2];
    std::vector<Complex> tables(static_cast<std::size_t>(n_images * stride));
    std::vector<double> occ_by_image(static_cast<std::size_t>(n_images));
    std::vector<int> form_by_image(static_cast<std::size_t>(n_images));

    for (py::ssize_t m = 0; m < n_ops; ++m) {
        const double* rot = rot_data + m * 9;
        const double* tran = tran_data + m * 3;
        for (py::ssize_t s = 0; s < n_sites; ++s) {
            const double* x = fract_data + s * 3;
            double image[3];
            for (int axis = 0; axis < 3; ++axis) {
                image[axis] = rot[axis * 3 + 0] * x[0] + rot[axis * 3 + 1] * x[1]
                            + rot[axis * 3 + 2] * x[2] + tran[axis];
            }
            const py::ssize_t p = m * n_sites + s;
            Complex* table = tables.data() + p * stride;
            fill_phase_table(image[0], low[0], high[0], table);
            fill_phase_table(image[1], low[1], high[1], table + span[0]);
            fill_phase_table(image[2], low[2], high[2], table + span[0] + span[1]);
            occ_by_image[static_cast<std::size_t>(p)] = occ_data[s];
            form_by_image[static_cast<std::size_t>(p)] = form_index_data[s];
        }
    }

    std::vector<char> is_aniso(static_cast<std::size_t>(n_sites), 0);
    bool any_aniso = false;
    for (py::ssize_t s = 0; s < n_sites; ++s) {
        const double* u = aniso_data + s * 6;
        for (int i = 0; i < 6; ++i) {
            if (u[i] != 0.0) {
                is_aniso[static_cast<std::size_t>(s)] = 1;
                any_aniso = true;
                break;
            }
        }
    }

    Complex* out = result.mutable_data();

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<double> weight(static_cast<std::size_t>(n_sites));

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (py::ssize_t n = 0; n < n_refl; ++n) {
            const int h = hkl_data[n * 3 + 0];
            const int k = hkl_data[n * 3 + 1];
            const int l = hkl_data[n * 3 + 2];
            const py::ssize_t ih = h - low[0];
            const py::ssize_t ik = span[0] + (k - low[1]);
            const py::ssize_t il = span[0] + span[1] + (l - low[2]);
            const double s2 = stol2_data[n];

            // occupancy * scattering factor, and for isotropic sites also the
            // Debye-Waller factor, which does not depend on the symmetry image.
            for (py::ssize_t s = 0; s < n_sites; ++s) {
                double value = occ_data[s]
                    * form_data[static_cast<py::ssize_t>(form_index_data[s]) * n_refl + n];
                if (!is_aniso[static_cast<std::size_t>(s)]) {
                    value *= std::exp(-kUtoB * s2 * uiso_data[s]);
                }
                weight[static_cast<std::size_t>(s)] = value;
            }

            Complex total(0.0, 0.0);
            for (py::ssize_t m = 0; m < n_ops; ++m) {
                double quad[6] = {0, 0, 0, 0, 0, 0};
                if (any_aniso) {
                    const double* rot = rot_data + m * 9;
                    // Miller indices transform as row vectors: h' = h . R.
                    const double hx = (h * rot[0] + k * rot[3] + l * rot[6]) * reciprocal[0];
                    const double hy = (h * rot[1] + k * rot[4] + l * rot[7]) * reciprocal[1];
                    const double hz = (h * rot[2] + k * rot[5] + l * rot[8]) * reciprocal[2];
                    quad[0] = hx * hx;
                    quad[1] = hy * hy;
                    quad[2] = hz * hz;
                    quad[3] = 2.0 * hx * hy;
                    quad[4] = 2.0 * hx * hz;
                    quad[5] = 2.0 * hy * hz;
                }
                const Complex* base = tables.data() + m * n_sites * stride;
                for (py::ssize_t s = 0; s < n_sites; ++s) {
                    double value = weight[static_cast<std::size_t>(s)];
                    if (is_aniso[static_cast<std::size_t>(s)]) {
                        const double* u = aniso_data + s * 6;
                        const double r_u_r = quad[0] * u[0] + quad[1] * u[1]
                                           + quad[2] * u[2] + quad[3] * u[3]
                                           + quad[4] * u[4] + quad[5] * u[5];
                        value *= std::exp(-kTwoPiSq * r_u_r);
                    }
                    const Complex* table = base + s * stride;
                    total += value * (table[ih] * table[ik] * table[il]);
                }
            }
            out[n] = total;
        }
    }

    return result;
}

}  // namespace

PYBIND11_MODULE(density_cpp, m) {
    m.doc() = R"doc(
Fast C++ implementation of classic Marching Cubes isosurface extraction for
regular 3-D scalar density grids.

Build with:
    uv pip install pybind11
    uv pip install -e . --no-build-isolation

The module is intentionally optional. Downstream Python code should import it
with try/except ImportError and present a clear message when the compiled
extension has not been built.
)doc";

    m.def(
        "marching_cubes",
        &marching_cubes,
        py::arg("grid"),
        py::arg("level"),
        py::arg("origin") = std::array<double, 3>{0.0, 0.0, 0.0},
        py::arg("step") = std::array<double, 3>{1.0, 1.0, 1.0},
        py::arg("mask") = py::none(),
        py::arg("block") = 8,
        R"doc(
Extract a wireframe isosurface from a regular 3-D scalar density grid.

Parameters
----------
grid : numpy.ndarray
    Three-dimensional C-contiguous float32 or float64 array with shape
    (nx, ny, nz).
level : float
    Isosurface level.
origin : tuple[float, float, float], optional
    Cartesian coordinates of grid[0, 0, 0].
step : tuple[float, float, float], optional
    Cartesian spacing along the x, y, and z grid axes.
mask : numpy.ndarray or None, optional
    Coarse C-contiguous boolean occupancy mask over blocks of cubes, with
    shape ceil((nx - 1) / block), ceil((ny - 1) / block),
    ceil((nz - 1) / block). Cubes in a block whose entry is False are
    skipped and their density is never read. None (the default) contours
    the whole grid.
block : int, optional
    Number of cubes per mask entry along each axis. Ignored when mask is
    None.

Returns
-------
(vertices, edges) : tuple[numpy.ndarray, numpy.ndarray]
    vertices is a float64 array with shape (M, 3).
    edges is an int64 array with shape (K, 2) containing unique undirected
    vertex-index pairs suitable for GL_LINES-style rendering.

Notes
-----
Vertices are shared across neighbouring cubes by hashing the canonical identity
of the intersected grid edge. Empty or fully solid grids return zero-row output
arrays with the documented dtypes and shapes.

The mask only restricts which cubes are visited; within the cubes it selects
the result is identical to an unmasked run, and the cubes are visited in the
same order, so the vertices are numbered as an unmasked scan would number
them.
)doc"
    );

    m.def(
        "structure_factors",
        &structure_factors,
        py::arg("hkl"),
        py::arg("stol2"),
        py::arg("rotations"),
        py::arg("translations"),
        py::arg("fract"),
        py::arg("occupancies"),
        py::arg("u_iso"),
        py::arg("aniso"),
        py::arg("form_index"),
        py::arg("form_factors"),
        py::arg("reciprocal"),
        R"doc(
Sum calculated structure factors over a small-molecule model.

Implements the same formula as gemmi's StructureFactorCalculator, but
tabulates the separable phase factor exp(2 pi i (hx + ky + lz)) over the
Miller-index range of the data, so the inner loop costs two complex
multiplications instead of a sine and a cosine.

Parameters
----------
hkl : numpy.ndarray
    (N, 3) int32 Miller indices.
stol2 : numpy.ndarray
    (N,) float64 array of (sin(theta) / lambda)^2.
rotations : numpy.ndarray
    (M, 3, 3) float64 symmetry rotation matrices, identity included.
translations : numpy.ndarray
    (M, 3) float64 symmetry translations, matching `rotations`.
fract : numpy.ndarray
    (S, 3) float64 fractional coordinates of the sites.
occupancies, u_iso : numpy.ndarray
    (S,) float64 site occupancies and isotropic ADPs.
aniso : numpy.ndarray
    (S, 6) float64 anisotropic ADPs as U11, U22, U33, U12, U13, U23 in the
    small-molecule (CIF) convention. An all-zero row means the site is
    isotropic and `u_iso` is used instead.
form_index : numpy.ndarray
    (S,) int32 row of `form_factors` to use for each site.
form_factors : numpy.ndarray
    (F, N) float64 scattering factor of every distinct scatterer for every
    reflection, including any addend such as f'.
reciprocal : tuple[float, float, float]
    Reciprocal cell lengths a*, b*, c*, used for the anisotropic
    Debye-Waller factor.

Returns
-------
numpy.ndarray
    (N,) complex128 array of calculated structure factors.
)doc"
    );
}
