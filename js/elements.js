/**
 * Element metadata (colours, covalent radii) ported from
 * `fastmolwidget/atoms.py`. Kept as a standalone module so it can be reused
 * or replaced without touching the renderer.
 */

export const ELEMENT2COLOR = {
  H: '#FFFFFF', He: '#FFFFFF', Li: '#CC80FF', Be: '#c9d5e9', B: '#FFB5B5',
  C: '#797979', N: '#3050F8', O: '#FF0D0D', F: '#90e001', Ne: '#B3E3F5',
  Na: '#AB5CF2', Mg: '#bbc7db', Al: '#BFA6A6', Si: '#F0C8A0', P: '#FF8000',
  S: '#eeee2c', Cl: '#419941', Ar: '#80D1E3', K: '#8F40D4', Ca: '#bbc7db',
  Sc: '#E6E6E6', Ti: '#BFC2C7', V: '#A6A6AB', Cr: '#8A99C7', Mn: '#9C7AC7',
  Fe: '#E06633', Co: '#F090A0', Ni: '#50D050', Cu: '#C88033', Zn: '#7D80B0',
  Ga: '#C28F8F', Ge: '#668F8F', As: '#BD80E3', Se: '#FFA100', Br: '#A62929',
  Kr: '#5CB8D1', Rb: '#702EB0', Sr: '#bbc7db', Y: '#94FFFF', Zr: '#94E0E0',
  Nb: '#73C2C9', Mo: '#54B5B5', Tc: '#3B9E9E', Ru: '#248F8F', Rh: '#0A7D8C',
  Pd: '#006985', Ag: '#C0C0C0', Cd: '#FFD98F', In: '#A67573', Sn: '#668080',
  Sb: '#9E63B5', Te: '#D47A00', I: '#940094', Xe: '#429EB0', Cs: '#57178F',
  Ba: '#bbc7db', La: '#d9ffff', Ce: '#d9ffff', Pr: '#d9ffff', Nd: '#d9ffff',
  Pm: '#d9ffff', Sm: '#d9ffff', Eu: '#d9ffff', Gd: '#d9ffff', Tb: '#d9ffff',
  Dy: '#d9ffff', Ho: '#d9ffff', Er: '#d9ffff', Tm: '#d9ffff', Yb: '#d9ffff',
  Lu: '#d9ffff', Hf: '#d9ffff', Ta: '#d9ffff', W: '#d9ffff', Re: '#d9ffff',
  Os: '#d9ffff', Ir: '#d9ffff', Pt: '#d9ffff', Au: '#d9ffff', Hg: '#d9ffff',
  Tl: '#d9ffff', Pb: '#d9ffff', Bi: '#d9ffff', Po: '#d9ffff', At: '#d9ffff',
  Rn: '#d9ffff', Fr: '#d9ffff', Ra: '#d9ffff', Ac: '#d9ffff', Th: '#d9ffff',
  Pa: '#d9ffff', U: '#d9ffff', Np: '#d9ffff', Pu: '#d9ffff', Am: '#d9ffff',
  Cm: '#d9ffff', Bk: '#d9ffff', Cf: '#d9ffff', D: '#e2e6e6',
};

export const ELEMENT2COV = {
  H: 0.50, He: 0.50, Li: 1.23, Be: 0.9, B: 0.82, C: 0.77, N: 0.75, O: 0.73,
  F: 0.72, Ne: 0.71, Na: 1.54, Mg: 1.36, Al: 1.18, Si: 1.11, P: 1.06,
  S: 1.02, Cl: 0.99, Ar: 0.98, K: 2.03, Ca: 1.74, Sc: 1.44, Ti: 1.32,
  V: 1.22, Cr: 1.18, Mn: 1.17, Fe: 1.17, Co: 1.16, Ni: 1.15, Cu: 1.17,
  Zn: 1.25, Ga: 1.26, Ge: 1.22, As: 1.2, Se: 1.16, Br: 1.14, Kr: 1.12,
  Rb: 2.16, Sr: 1.91, Y: 1.62, Zr: 1.45, Nb: 1.34, Mo: 1.3, Tc: 1.27,
  Ru: 1.25, Rh: 1.25, Pd: 1.28, Ag: 1.34, Cd: 1.48, In: 1.44, Sn: 1.41,
  Sb: 1.4, Te: 1.36, I: 1.33, Xe: 1.31, Cs: 2.35, Ba: 1.98, La: 1.69,
  Ce: 1.65, Pr: 1.65, Nd: 1.64, Pm: 1.63, Sm: 1.62, Eu: 1.85, Gd: 1.61,
  Tb: 1.59, Dy: 1.59, Ho: 1.58, Er: 1.57, Tm: 1.56, Yb: 1.74, Lu: 1.56,
  Hf: 1.44, Ta: 1.34, W: 1.3, Re: 1.28, Os: 1.26, Ir: 1.27, Pt: 1.3,
  Au: 1.34, Hg: 1.49, Tl: 1.48, Pb: 1.47, Bi: 1.46, Po: 1.46, At: 1.45,
  Rn: 1.0, Fr: 1.0, Ra: 1.0, Ac: 1.88, Th: 1.65, Pa: 1.61, U: 1.42,
  Np: 1.30, Pu: 1.51, Am: 1.82, Cm: 1.20, Bk: 1.20, Cf: 1.20, D: 0.5,
};

const KNOWN_ATOMS = new Set(Object.keys(ELEMENT2COV));

/** Port of `atoms.get_atomlabel`: strips trailing digits/noise from a SHELX
 * style atom name, e.g. 'C12' -> 'C', 'Ca1' -> 'Ca'. Falls back to the raw
 * (capitalised) input when it cannot be resolved. */
export function getAtomLabel(inputAtom) {
  let atom = '';
  for (const ch of inputAtom) {
    if (/^[A-Za-z#]/.test(ch)) {
      atom += ch.toUpperCase();
    } else {
      break;
    }
  }
  if (!atom) return inputAtom;
  const two = atom.slice(0, 2);
  const twoCap = two.charAt(0) + two.charAt(1).toLowerCase();
  if (KNOWN_ATOMS.has(twoCap)) return twoCap;
  const one = atom.charAt(0);
  if (KNOWN_ATOMS.has(one)) return one;
  return atom;
}

export function getRadiusFromElement(element) {
  const cleaned = getAtomLabel(element);
  const cap = cleaned.charAt(0) + cleaned.slice(1).toLowerCase();
  return ELEMENT2COV[cap] ?? ELEMENT2COV.C;
}

export function getElementColor(element) {
  const cap = element.charAt(0) + element.slice(1).toLowerCase();
  return ELEMENT2COLOR[cap] ?? '#000000';
}
