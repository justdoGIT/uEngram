import numpy as np
import pytest

from uengram.host import schema
from uengram.host.features import windows
from uengram.host.synth import make_ticks


def test_every_measured_field_has_exactly_one_owner_and_one_modality():
    schema.validate()
    owner = schema.owned_fields()
    by_name = {a.name: a for a in schema.ATTRIBUTES}
    for f, (_, role) in schema.TICK_FIELDS.items():
        if role in ("counter", "gauge") and f in owner:
            assert by_name[owner[f]].modality in schema.MODALITIES
    # an excluded signal never re-enters as an attribute
    assert not set(schema.EXCLUDED) & {a.name for a in schema.ATTRIBUTES}


def test_attribute_owned_twice_is_rejected(monkeypatch):
    dup = schema.Attribute("cpu_busy_again", "memory", "fraction", ("cpu_busy_j",), "cpu_total_j")
    monkeypatch.setattr(schema, "ATTRIBUTES", schema.ATTRIBUTES + (dup,))
    with pytest.raises(ValueError, match="owned by"):
        schema.validate()


def test_no_feature_is_a_deterministic_function_of_the_others():
    """The Jacobian of the Gaussian features with respect to every measured
    (field, tick) input has full row rank: no feature (attribute level or
    burstiness) can be computed from the others."""
    base = make_ticks(32, seed=3, order=("build",))
    names = schema.feature_names()
    keep = [i for i, n in enumerate(names) if n != "thermal_residual"]  # filled by the encoder
    fields = [f for f, (_, role) in schema.TICK_FIELDS.items() if role in ("counter", "gauge")]
    copies = [[dict(r, t=r["t"] + 32 * j) for r in base] for j in range(1 + len(fields) * 32)]
    j = 1
    for f in fields:
        for k in range(32):  # copy j perturbs one input: field f at tick k
            v = copies[j][k][f]
            copies[j][k][f] = v * (1.0 + 1e-4) if v else 1e-4
            j += 1
    F = windows([r for c in copies for r in c], 32).F[:, keep]
    J = F[1:] - F[0]
    assert np.all(np.isfinite(J))
    assert np.linalg.matrix_rank(J.T, tol=1e-10) == len(keep)
