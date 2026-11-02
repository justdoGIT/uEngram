import math

import numpy as np
import pytest

from uengram.host.events import EventModel, betainc, nb_midp


def _ibeta_int_b(a: float, n: int, x: float) -> float:
    """Exact I_x(a, n) for integer n: x^a sum_{k<n} Gamma(a+k)/(Gamma(a) k!) (1-x)^k."""
    return x ** a * sum(math.exp(math.lgamma(a + k) - math.lgamma(a) - math.lgamma(k + 1)) * (1 - x) ** k
                        for k in range(n))


@pytest.mark.parametrize("a,n,x", [(1.0, 1, 0.3), (2.5, 1, 0.7), (0.5, 3, 0.05), (40.0, 7, 0.9),
                                   (3.5, 12, 0.6)])
def test_betainc_matches_closed_form(a, n, x):
    assert betainc(a, n, x) == pytest.approx(_ibeta_int_b(a, n, x), rel=1e-10)
    assert betainc(n, a, 1 - x) == pytest.approx(1 - _ibeta_int_b(a, n, x), rel=1e-9, abs=1e-14)


def test_nb_midp_matches_pmf_sum():
    r, q = 3.5, 0.9
    k = np.arange(0, 400)
    logpmf = np.array([math.lgamma(j + r) - math.lgamma(j + 1) - math.lgamma(r) + r * math.log(q)
                       + j * math.log1p(-q) for j in k])
    pmf = np.exp(logpmf)
    for x in (0, 2, 7, 30):
        assert nb_midp(x, r, q) == pytest.approx(pmf[x + 1:].sum() + 0.5 * pmf[x], rel=1e-9)


def test_first_ever_error_alarms_and_background_rate_does_not():
    E = np.zeros((2700, 4))
    E[:, 2] = np.random.default_rng(0).poisson(3.0, 2700)  # NIC drops at a steady background rate
    ev = EventModel().fit(E)
    p_quiet = ev.pvalues(np.array([0, 0, 3, 0]))
    assert np.all(p_quiet > 0.1)
    p = ev.pvalues(np.array([1, 0, 3, 0]))  # one OOM kill after a day without any
    assert p[0] < 0.01 / 2 / 4
    p_burst = ev.pvalues(np.array([0, 0, 15, 0]))
    assert p_burst[2] < 1e-5
