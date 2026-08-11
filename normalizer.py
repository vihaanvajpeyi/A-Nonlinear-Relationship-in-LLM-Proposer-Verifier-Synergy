"""
normalizer.py — Answer normalizer for the synergy prediction experiment.
Version: 2.1 (clean rewrite)

Handles: numeric, fraction, decimal, symbolic (CAS), boolean, multi-value,
         string fallback, and code grading.
"""

import ast
import math
import re
import signal
import unicodedata
from collections import Counter
from fractions import Fraction


# ---------------------------------------------------------------------------
# 0. Unicode + LaTeX normalization
# ---------------------------------------------------------------------------

def _normalize_unicode(s: str) -> str:
    """Normalize Unicode and LaTeX wrappers to plain ASCII."""
    s = unicodedata.normalize('NFC', s)
    s = s.replace('\u2212', '-')    # Unicode minus
    s = s.replace('\u00d7', '*')    # ×
    s = s.replace('\u00b7', '*')    # middle dot
    s = s.replace('\u2019', "'")    # right single quote
    s = s.replace('\u201c', '"').replace('\u201d', '"')
    # Strip LaTeX: \boxed{x} -> x, \text{x} -> x
    s = re.sub(r'\\boxed\{([^}]+)\}', lambda m: m.group(1), s)
    s = re.sub(r'\\text\{([^}]+)\}',  lambda m: m.group(1), s)
    # Strip bare $ signs (inline math delimiters)
    s = s.replace('$$', '').replace('$', '')
    return s


# ---------------------------------------------------------------------------
# 1. Label=value extraction  e.g. "Area = 49π sq units." -> "49π sq units"
# ---------------------------------------------------------------------------

def _extract_math_value(s: str) -> str:
    """
    If s looks like 'Label = <value>', extract just the value.
    e.g. 'Area = 49π square units.' -> '49π square units.'
    """
    s = s.strip().rstrip('.')
    m = re.match(r'^[A-Za-z][A-Za-z ]*=[ ]*(.+)$', s)
    if m:
        return m.group(1).strip()
    return s


# ---------------------------------------------------------------------------
# 2. Currency / unit stripping
# ---------------------------------------------------------------------------

_STRIP_PATTERN = re.compile(
    r'[$£€¥₹%]'
    r'|\b(mph|km/h|m/s|kg|g|mg|km|cm|mm'
    r'|liters?|litres?|ml|gallons?'
    r'|hours?|hrs?|minutes?|mins?|seconds?|secs?'
    r'|cents?|dollars?|years?|days?|weeks?'
    r'|square units?|cubic units?)\b',
    re.IGNORECASE
)


# ---------------------------------------------------------------------------
# 3. Symbolic CAS comparison
# ---------------------------------------------------------------------------

def _sympy_available():
    try:
        import sympy
        return True
    except ImportError:
        return False


_MATH_PATTERN = re.compile(
    r'[\d+\-*/^√π]|sqrt|log|sin|cos|tan|\bpi\b', re.IGNORECASE
)


def _has_math_content(s: str) -> bool:
    return bool(_MATH_PATTERN.search(s))


def _prep_symbolic(s: str) -> str:
    s = _normalize_unicode(s)
    s = _extract_math_value(s)
    s = _STRIP_PATTERN.sub('', s).strip().rstrip('.')
    s = s.strip()
    # N√M -> N*sqrt(M)
    s = re.sub(r'(\d+)\s*√\s*(\d+)', r'\1*sqrt(\2)', s)
    s = re.sub(r'√\s*(\d+)', r'sqrt(\1)', s)
    # Nπ -> N*pi
    s = re.sub(r'(\d)\s*π', r'\1*pi', s)
    s = s.replace('π', 'pi')
    return s


def _symbolic_equal(a: str, b: str) -> 'bool | None':
    if not _sympy_available():
        return None
    if not _has_math_content(a) and not _has_math_content(b):
        return None
    try:
        from sympy.parsing.sympy_parser import (
            parse_expr, standard_transformations,
            implicit_multiplication_application,
        )
        import sympy
        T = standard_transformations + (implicit_multiplication_application,)
        ea = parse_expr(_prep_symbolic(a), transformations=T)
        eb = parse_expr(_prep_symbolic(b), transformations=T)
        if ea.is_Symbol and eb.is_Symbol:
            return None
        return sympy.simplify(ea - eb) == 0
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 4. Numeric comparison
# ---------------------------------------------------------------------------

def _to_number(s: str) -> 'float | None':
    s = _normalize_unicode(s).strip()
    s = _extract_math_value(s)
    s = s.replace(',', '')
    s = _STRIP_PATTERN.sub('', s).strip().rstrip('.')
    if not s:
        return None
    if s.lower() in ('nan', 'inf', 'infinity', '-inf', '-infinity'):
        return float(s)
    try:
        return float(s)
    except ValueError:
        pass
    try:
        return float(Fraction(s))
    except (ValueError, ZeroDivisionError):
        pass
    return None


def _numeric_equal(a: str, b: str) -> 'bool | None':
    na, nb = _to_number(a), _to_number(b)
    if na is None or nb is None:
        return None
    if math.isnan(na) and math.isnan(nb):
        return True
    if math.isinf(na) or math.isinf(nb):
        return na == nb
    return math.isclose(na, nb, rel_tol=1e-9, abs_tol=1e-4)


# ---------------------------------------------------------------------------
# 5. Boolean
# ---------------------------------------------------------------------------

_TRUE_TOKENS  = {'true', 'yes', '1'}
_FALSE_TOKENS = {'false', 'no', '0'}


def _as_bool(s: str) -> 'bool | None':
    t = s.strip().lower()
    if t in _TRUE_TOKENS:  return True
    if t in _FALSE_TOKENS: return False
    return None


def _bool_equal(a: str, b: str) -> 'bool | None':
    ba, bb = _as_bool(a), _as_bool(b)
    if ba is None or bb is None:
        return None
    return ba == bb


# ---------------------------------------------------------------------------
# 6. Multi-value (unordered multiset)
# ---------------------------------------------------------------------------

def _split_multi(s: str) -> list:
    parts = re.split(r'\band\b|,|;', s, flags=re.IGNORECASE)
    cleaned = []
    for p in parts:
        p = p.strip()
        p = re.sub(r'^[a-zA-Z_]\w*\s*=\s*', '', p)
        if p:
            cleaned.append(p)
    return cleaned


def _atomic_key(p: str):
    n = _to_number(p)
    if n is not None:
        return ('num', round(n, 9))
    if _sympy_available() and _has_math_content(p):
        try:
            from sympy.parsing.sympy_parser import (
                parse_expr, standard_transformations,
                implicit_multiplication_application,
            )
            import sympy
            T = standard_transformations + (implicit_multiplication_application,)
            expr = parse_expr(_prep_symbolic(p), transformations=T)
            return ('sym', str(sympy.simplify(expr)))
        except Exception:
            pass
    return ('str', _normalize_str(p))


def _multi_equal(a: str, b: str) -> 'bool | None':
    pa, pb = _split_multi(a), _split_multi(b)
    if len(pa) != len(pb) or len(pa) < 2:
        return None
    try:
        return Counter(_atomic_key(p) for p in pa) == Counter(_atomic_key(p) for p in pb)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 7. String fallback
# ---------------------------------------------------------------------------

def _normalize_str(s: str) -> str:
    s = _normalize_unicode(s)
    s = unicodedata.normalize('NFC', s).strip().lower()
    s = re.sub(r'\s+', ' ', s)
    s = s.rstrip('.')
    return s


# ---------------------------------------------------------------------------
# 8. Empty check
# ---------------------------------------------------------------------------

def _is_empty(s) -> bool:
    if s is None: return True
    return str(s).strip() == ''


# ---------------------------------------------------------------------------
# 9. Main entry point
# ---------------------------------------------------------------------------

def answers_match(predicted: str, ground_truth: str) -> bool:
    """
    Returns True if predicted and ground_truth represent the same answer.
    Pipeline: empty → boolean → numeric → symbolic → multi-value → string
    Never raises.
    """
    if _is_empty(predicted) or _is_empty(ground_truth):
        return False

    p = str(predicted).strip()
    g = str(ground_truth).strip()

    r = _bool_equal(p, g)
    if r is not None: return r

    r = _numeric_equal(p, g)
    if r is not None: return r

    r = _symbolic_equal(p, g)
    if r is not None: return r

    _has_sep = lambda s: bool(re.search(r'\band\b|,|;|=', s))
    if _has_sep(p) and _has_sep(g):
        r = _multi_equal(p, g)
        if r is not None: return r

    return _normalize_str(p) == _normalize_str(g)


# ---------------------------------------------------------------------------
# 10. Code grader
# ---------------------------------------------------------------------------

_SAFE_BUILTINS = {
    'abs','all','any','bin','bool','chr','dict','divmod','enumerate',
    'filter','float','frozenset','getattr','hasattr','hash','hex','int',
    'isinstance','issubclass','iter','len','list','map','max','min',
    'next','oct','ord','pow','print','range','repr','reversed','round',
    'set','setattr','slice','sorted','str','sum','tuple','type','zip',
    'True','False','None',
}
import collections as _collections, itertools as _itertools, math as _math
import builtins as _builtins_module
_SAFE_GLOBALS = {k: getattr(_builtins_module, k)
                 for k in _SAFE_BUILTINS if hasattr(_builtins_module, k)}
_SAFE_GLOBALS.update({
    '__builtins__': _SAFE_GLOBALS,
    'collections': _collections,
    'itertools': _itertools,
    'math': _math,
})


def _timeout_handler(signum, frame):
    raise TimeoutError("Code execution timed out")


def grade_code(function_str: str, test_cases: list, expected_func_name: str = None,
               timeout_seconds: int = 5) -> float:
    if _is_empty(function_str) or not test_cases:
        return 0.0
    namespace = dict(_SAFE_GLOBALS)
    try:
        exec(compile(function_str, '<model_output>', 'exec'), namespace)
    except Exception:
        return 0.0
    # Find target function by name, then AST fallback
    func = namespace.get(expected_func_name) if expected_func_name else None
    if func is None:
        try:
            tree = ast.parse(function_str)
            top = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
            if top:
                func = namespace.get(top[-1].name)
        except Exception:
            pass
    if func is None:
        m = re.search(r'def\s+(\w+)\s*\(', function_str)
        if m:
            func = namespace.get(m.group(1))
    if func is None:
        return 0.0
    passed = 0
    for tc in test_cases:
        try:
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout_seconds)
            try:
                result = eval(f"func({tc['input']})", {'func': func})
                if result == eval(tc['expected']):
                    passed += 1
            finally:
                signal.alarm(0)
        except Exception:
            pass
    return passed / len(test_cases)


# ---------------------------------------------------------------------------
# 11. Self-tests
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    tests = [
        # Numeric
        ("3.0",         "3",           True),
        ("0.5",         "1/2",         True),
        ("-4",          "-4",          True),
        ("10.30",       "10.3",        True),
        ("5/33",        "0.15152",     True),
        ("1e-6",        "0.000001",    True),
        ("2.5e3",       "2500",        True),
        ("1000000",     "1000000.00005", True),
        # Currency / units
        ("$3.75",       "3.75",        True),
        ("$4,680",      "4680",        True),
        ("-4%",         "-4",          True),
        ("-$202.50",    "-202.50",     True),
        ("$5,200",      "5200",        True),
        ("10.5 hours",  "10.5",        True),
        ("96 mph",      "96",          True),
        # Label=value + units
        ("Area = 49π square units.", "49π",    True),
        ("Balance = $1159.69",       "1159.69", True),
        # LaTeX
        (r"\boxed{17/6}",  "17/6",    True),
        (r"\boxed{720}",   "720",     True),
        (r"\boxed{4}",     "4",       True),
        # Symbolic
        ("25√3",        "25*sqrt(3)",  True),
        ("49π",         "49*pi",       True),
        ("96π",         "96*pi",       True),
        ("sqrt(75)",    "5*sqrt(3)",   True),
        ("4/5",         "0.8",         True),
        # Boolean
        ("True",        "true",        True),
        ("False",       "false",       True),
        ("yes",         "true",        True),
        ("True",        "False",       False),
        ("correct",     "true",        False),
        # Multi-value ordered
        ("x=1, y=5",    "x=1, y=5",   True),
        ("3 and 1/2",   "3 and 0.5",  True),
        # Multi-value REORDERED
        ("1/2 and 3",   "3 and 1/2",  True),
        ("y=5, x=1",    "x=1, y=5",   True),
        ("-2 and 7",    "7 and -2",    True),
        ("1 and 2",     "1 and 3",     False),
        # String fallback
        ("TypeError",   "typeerror",   True),
        ("[1,2,3,4]",   "[1, 2, 3, 4]", True),
        ("total += x",  "total += x",  True),
        ("(lo + hi) // 2", "(lo + hi) // 2", True),
        # Empty
        (None,          "42",          False),
        ("",            "42",          False),
        ("   ",         "42",          False),
        # Should be False
        ("42",          "43",          False),
        ("1/6",         "1/7",         False),
        ("25√3",        "25√2",        False),
        # NaN / Inf
        ("nan",         "nan",         True),
        ("inf",         "inf",         True),
        ("inf",         "-inf",        False),
        # M047 rounding fix
        ("1.96",        "1.96",        True),
        ("1.97",        "1.96",        False),
        # pi detection
        ("pi",          "pi",          True),
    ]

    print("Running normalizer self-tests...\n")
    passed_n = failed_n = 0
    for predicted, ground_truth, expected in tests:
        result = answers_match(predicted, ground_truth)
        if result == expected:
            passed_n += 1
        else:
            print(f"  FAIL: answers_match({predicted!r}, {ground_truth!r})")
            print(f"        got={result}, expected={expected}")
            failed_n += 1

    print(f"\n{passed_n}/{passed_n+failed_n} normalizer tests passed.")

    # Code grader tests
    print("\nRunning code grader tests...\n")
    cg_tests = [
        ("def add(a,b):\n    return a+b",
         [{"input":"1,2","expected":"3"},{"input":"0,0","expected":"0"}],
         "add", 1.0),
        ("def helper(x):\n    return x*2\ndef solution(lst):\n    return [helper(x) for x in lst]",
         [{"input":"[1,2,3]","expected":"[2,4,6]"}],
         "solution", 1.0),
        ("def bad(x):\n    return 1/0",
         [{"input":"1","expected":"1"}],
         "bad", 0.0),
        ("", [{"input":"1","expected":"1"}], None, 0.0),
        ("def f(x):\n    return x*2",
         [{"input":"2","expected":"4"},{"input":"3","expected":"9"}],
         "f", 0.5),
    ]
    cg_passed = cg_failed = 0
    for code, tcs, fname, expected_score in cg_tests:
        result = grade_code(code, tcs, expected_func_name=fname)
        if abs(result - expected_score) < 1e-9:
            cg_passed += 1
        else:
            print(f"  FAIL: grade_code expected={expected_score}, got={result}")
            cg_failed += 1

    print(f"{cg_passed}/{cg_passed+cg_failed} code grader tests passed.")
    total = passed_n + cg_passed
    total_all = passed_n + failed_n + cg_passed + cg_failed
    print(f"\nTotal: {total}/{total_all} passed.")
    if failed_n == 0 and cg_failed == 0:
        print("All tests passed — normalizer v2.1 is ready.")
