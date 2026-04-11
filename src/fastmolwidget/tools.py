from __future__ import annotations


def to_float(st) -> float | list[float] | None:
    if isinstance(st, list):
        try:
            return [float(x) for x in st[-2:]]
        except ValueError:
            return None
    else:
        try:
            return float(st.split('(')[0])
        except ValueError:
            return None


def get_error_from_value(value: str) -> tuple[float, float]:
    """
    Returns the error value from a number string.
    """
    try:
        value = value.replace(" ", "")
    except AttributeError:
        return float(value), 0.0
    if "(" in value:
        vval, err = value.split("(")
        val = vval.split('.')
        err = err.split(")")[0]
        if not err:
            return float(vval), 0.0
        if len(val) > 1:
            return float(vval), int(err) * (10 ** (-1 * len(val[1])))
        else:
            return float(vval), float(err)
    else:
        try:
            return float(value), 0.0
        except ValueError:
            return 0.0, 0.0


def isnumeric(value: str) -> bool:
    """
    Determines if a string can be converted to a number.
    """
    value = value.split('(')[0]
    try:
        float(value)
    except ValueError:
        return False
    return True


def grouper(inputs, n, fillvalue=None):
    iters = [iter(inputs)] * n
    return it.zip_longest(*iters, fillvalue=fillvalue)
