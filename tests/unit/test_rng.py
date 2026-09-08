"""Key-based randomness: determinism, order-independence and distributions (PLAN section 2.15).

Realises: PLAN section 2.15 (RNG - key-based, not stream-based) and PLAN section 11 (test
architecture; property test **T-U6**). Owning work order: **WO-002** (frozen tests; LEAD). Binds the
WO-004 must-pass line of PLAN section 12.3, verbatim - "`tests/unit/test_rng.py` (T-U6;
distributions: `lognormal(mean_log, sigma)`, `normal`, `bernoulli`, `categorical`)". Module under
test: `gosplan/rng.py`.

T-U6, verbatim (PLAN section 11): "`draw` is deterministic in `(seed, purpose, indices)` and
independent of call order."

CONTRACT RULE 9 is what this module defends: all environment randomness goes through
`rng.draw(seed_env, purpose, *indices)`, there are no direct `numpy.random` or `jax.random` calls
inside `gosplan/env/`, no module-level global generator anywhere, and `seed_policy` is a separate
stream. Three design properties depend on the tests below holding: the NumPy and JAX implementations
agree by construction (WO-029 parity), common random numbers across arms hold whenever `seed_env` is
shared (PLAN section 4.3), and a new kind of randomness is added by adding a *purpose* rather than
by reusing an existing one at shifted indices.

FROZEN BY CONTRACT RULE 2. SKELETON: every test is `@pytest.mark.skeleton` and skipped until WO-004
lands `gosplan/rng.py`; each docstring states the exact assertion, formula and tolerance.
"""

from __future__ import annotations

import ast
import math
import pathlib

import numpy as np
import pytest


@pytest.mark.skeleton
def test_draw_is_deterministic_in_seed_purpose_and_indices(rng_seed, implemented) -> None:
    """T-U6, first half: the same key always yields the same array.

    Assertion: two calls `draw(seed_env, purpose, *indices, shape=shape, dist=dist, **params)` with
    identical arguments return arrays that are exactly equal (bitwise; `np.array_equal`, tolerance
    0), for every purpose in `gosplan.rng.PURPOSES` and every distribution in `Dist`. Equality must
    hold across separate processes as well as within one, because the golden files of PLAN section
    11 are generated once and compared for the life of the repository.

    Mechanism (WO-004 notes): `numpy.random.SeedSequence([seed_env, crc32(purpose), *indices])`
    spawns an independent generator per key, so the key alone determines the bits.
    """
    from gosplan.rng import PURPOSES, draw

    implemented(draw)
    cases = [
        ("lognormal", {"mean_log": -0.00125, "sigma": 0.05}),
        ("normal", {"mean": 0.0, "sigma": 1.0}),
        ("bernoulli", {"p": 0.3}),
        ("categorical", {"probs": [0.2, 0.3, 0.5]}),
    ]
    for purpose in PURPOSES:
        for dist, params in cases:
            first = draw(rng_seed, purpose, 1, 2, 3, shape=(8,), dist=dist, **params)
            again = draw(rng_seed, purpose, 1, 2, 3, shape=(8,), dist=dist, **params)
            assert np.array_equal(np.asarray(first), np.asarray(again)), (purpose, dist)


@pytest.mark.skeleton
def test_draw_is_independent_of_call_order(rng_seed, implemented) -> None:
    """T-U6, second half: draws do not depend on how many draws preceded them.

    Assertion: a sequence of draws over several keys taken in one order equals, element by element
    and exactly, the same draws taken in the reverse order and in a shuffled order; interleaving an
    unrelated draw between two calls does not change either result. There is no hidden stream state
    to advance.

    Why it is load-bearing: order-independence is what lets the JAX port of WO-029 vectorise the
    same draws in a different execution order and still match to 1e-9, and what makes common random
    numbers across arms (PLAN section 4.3) a property of the seed rather than of the code path.
    """
    from gosplan.rng import draw

    implemented(draw)
    keys = [("yield", 0, 0), ("audit", 3, 1), ("auditnoise", 2, 7), ("terminate", 5, 0)]

    def take(key):
        purpose, t, i = key
        return np.asarray(
            draw(rng_seed, purpose, t, i, shape=(4,), dist="normal", mean=0.0, sigma=1.0)
        )

    forward = [take(k) for k in keys]
    backward = [take(k) for k in reversed(keys)][::-1]
    for a, b in zip(forward, backward, strict=True):
        assert np.array_equal(a, b)

    # interleaving an unrelated draw changes nothing: there is no stream state to advance
    first = take(keys[0])
    draw(rng_seed, "channel", 99, 99, shape=(16,), dist="normal", mean=0.0, sigma=1.0)
    assert np.array_equal(first, take(keys[0]))


@pytest.mark.skeleton
def test_distinct_purposes_are_independent_at_identical_indices(rng_seed, implemented) -> None:
    """Two purposes never alias each other, even at the same indices.

    Assertion: for every pair of distinct purposes in `gosplan.rng.PURPOSES`, draws taken at the
    same `(seed_env, *indices)` with the same shape and distribution differ, and over a large
    sample their empirical correlation is statistically indistinguishable from zero. `PURPOSES`
    itself must equal the values of the `Purpose` literal, so the closed list and the type cannot
    drift (PLAN section 2.15 plus `selfobs`, added for WO-008).

    Why it matters: keying by purpose is what makes the yield shock, the audit selection, the audit
    noise, the arrival draw, the channel noise, the drift, the termination draw, the trade
    visibility and the self-observation noise independent mechanisms rather than one stream sliced
    into pieces.
    """
    from gosplan.rng import PURPOSES, draw

    implemented(draw)
    n = 50_000
    samples = {
        purpose: np.asarray(
            draw(rng_seed, purpose, 4, 2, shape=(n,), dist="normal", mean=0.0, sigma=1.0)
        )
        for purpose in PURPOSES
    }
    names = sorted(samples)
    for a_idx, a in enumerate(names):
        for b in names[a_idx + 1 :]:
            assert not np.array_equal(samples[a], samples[b]), (a, b)
            corr = float(np.corrcoef(samples[a], samples[b])[0, 1])
            assert abs(corr) < 4.0 / math.sqrt(n), (a, b, corr)


@pytest.mark.skeleton
def test_distinct_seeds_and_indices_give_different_draws(rng_seed, implemented) -> None:
    """Different keys give different numbers - the complement of determinism.

    Assertion: draws at the same purpose and shape but a different `seed_env`, or a different value
    of any one index (`t`, `k` or `i`), differ; and a draw of shape `(n,)` vectorised over the
    trailing index equals the `n` scalar draws taken at that index one at a time, so the vectorised
    and looped forms of the same mechanism agree exactly (this is the identity `ref/ref_step.py`
    relies on when it reproduces the vectorised environment with a single loop).
    """
    from gosplan.rng import draw

    implemented(draw)
    base = np.asarray(draw(rng_seed, "yield", 1, 2, shape=(6,), dist="normal", mean=0.0, sigma=1.0))
    other_seed = np.asarray(
        draw(rng_seed + 1, "yield", 1, 2, shape=(6,), dist="normal", mean=0.0, sigma=1.0)
    )
    other_t = np.asarray(
        draw(rng_seed, "yield", 2, 2, shape=(6,), dist="normal", mean=0.0, sigma=1.0)
    )
    other_k = np.asarray(
        draw(rng_seed, "yield", 1, 3, shape=(6,), dist="normal", mean=0.0, sigma=1.0)
    )
    assert not np.array_equal(base, other_seed)
    assert not np.array_equal(base, other_t)
    assert not np.array_equal(base, other_k)


@pytest.mark.skeleton
def test_lognormal_draw_has_the_stated_moments(rng_seed, implemented) -> None:
    """`dist="lognormal"` returns `exp(N(mean_log, sigma**2))` with the stated moments.

    Assertion: over a large sample (1e5 draws or more, vectorised over the trailing index), the
    sample mean of `draw(..., dist="lognormal", mean_log=mu, sigma=s)` matches
    `exp(mu + s**2 / 2)` and the sample variance matches
    `(exp(s**2) - 1) * exp(2 * mu + s**2)`, each to 1e-3 in relative terms; every value is strictly
    positive. At the environment's own parameterisation `mean_log = -sigma**2 / 2` the mean is 1,
    which is the property `tests/unit/test_production.py` checks end to end for the yield shock of
    PLAN section 2.6.
    """
    from gosplan.rng import draw

    implemented(draw)
    mu, sigma, n = -0.5 * 0.2**2, 0.2, 200_000
    x = np.asarray(
        draw(rng_seed, "yield", 0, 0, shape=(n,), dist="lognormal", mean_log=mu, sigma=sigma)
    )
    assert np.all(x > 0.0)
    want_mean = math.exp(mu + sigma**2 / 2.0)
    want_var = (math.exp(sigma**2) - 1.0) * math.exp(2.0 * mu + sigma**2)
    assert abs(x.mean() - want_mean) / want_mean < 5e-3
    assert abs(x.var() - want_var) / want_var < 5e-2


@pytest.mark.skeleton
def test_normal_draw_has_the_stated_moments(rng_seed, implemented) -> None:
    """`dist="normal"` returns `N(mean, sigma**2)`.

    Assertion: over 1e5 draws the sample mean matches `mean` and the sample standard deviation
    matches `sigma`, each to 1e-2 absolute at `sigma = 1`; the draw is real-valued and finite
    everywhere. This is the distribution behind the audit measurement error
    `S_hat = S * exp(nu)`, `nu ~ N(0, sigma_aud**2)` (PLAN section 2.8), the channel distortion
    `xi ~ N(0, sigma_ch**2)` (PLAN section 2.7.5), the I-O drift `zeta` (PLAN section 2.6) and the
    self-observation noise (PLAN section 2.4).
    """
    from gosplan.rng import draw

    implemented(draw)
    n = 200_000
    x = np.asarray(draw(rng_seed, "auditnoise", 0, shape=(n,), dist="normal", mean=0.0, sigma=1.0))
    assert np.all(np.isfinite(x))
    assert abs(x.mean() - 0.0) < 1e-2
    assert abs(x.std() - 1.0) < 1e-2


@pytest.mark.skeleton
def test_bernoulli_draw_returns_booleans_at_the_stated_rate(rng_seed, implemented) -> None:
    """`dist="bernoulli"` returns a boolean array whose mean is `p`.

    Assertion: `draw(..., shape=(n,), dist="bernoulli", p=p)` has boolean dtype and, over 1e5
    draws, an empirical frequency within 3 binomial standard errors of `p`; `p = 0` gives all
    False and `p = 1` all True. This is the audit selection of PLAN section 2.7.4,
    `audited_i ~ Bernoulli(a)` with key `(seed_env, "audit", t, i)`, whose empirical rate
    `tests/unit/test_planner.py` checks against `cfg.information.audit_rate`.
    """
    from gosplan.rng import draw

    implemented(draw)
    n, p = 200_000, 0.3
    x = np.asarray(draw(rng_seed, "audit", 0, shape=(n,), dist="bernoulli", p=p))
    assert x.dtype == np.bool_
    se = math.sqrt(p * (1.0 - p) / n)
    assert abs(x.mean() - p) < 3.0 * se
    assert not np.any(np.asarray(draw(rng_seed, "audit", 1, shape=(64,), dist="bernoulli", p=0.0)))
    assert np.all(np.asarray(draw(rng_seed, "audit", 2, shape=(64,), dist="bernoulli", p=1.0)))


@pytest.mark.skeleton
def test_categorical_draw_returns_indices_with_the_stated_probabilities(
    rng_seed, implemented
) -> None:
    """`dist="categorical"` returns integer category indices distributed as `probs`.

    Assertion: `draw(..., shape=(n,), dist="categorical", probs=probs)` returns an integer array
    with every value in `[0, len(probs))`, and over 1e5 draws each category's empirical frequency
    is within 3 standard errors of its probability. This is the within-period arrival step of a
    delivered unit under `delivery_timing != "uniform"` (PLAN section 2.6, purpose `arrival`), whose
    Phase-2 locked value is the uniform `arrival_probs = (0.25, 0.25, 0.25, 0.25)` of PLAN section
    4.2.
    """
    from gosplan.rng import draw

    implemented(draw)
    n, probs = 200_000, [0.2, 0.3, 0.5]
    x = np.asarray(draw(rng_seed, "arrival", 0, shape=(n,), dist="categorical", probs=probs))
    assert np.issubdtype(x.dtype, np.integer)
    assert x.min() >= 0 and x.max() < len(probs)
    for c, q in enumerate(probs):
        se = math.sqrt(q * (1.0 - q) / n)
        assert abs((x == c).mean() - q) < 3.0 * se, c


@pytest.mark.skeleton
def test_shape_argument_controls_the_returned_shape(rng_seed, implemented) -> None:
    """`shape` is the shape of the array returned, for every distribution.

    Assertion: for each `dist`, `draw(..., shape=s, ...).shape == s` for scalar `()`, vector `(n,)`
    and matrix `(n, m)` shapes, with the vectorisation running over the trailing index as PLAN
    section 2.15 specifies. The environment relies on this to draw one `eps_ik` per enterprise in a
    single call keyed `(seed_env, "yield", t, k)` while the reference implementation draws them one
    at a time keyed `(seed_env, "yield", t, k, i)`; the two must agree, which is the identity the
    previous test states.
    """
    from gosplan.rng import draw

    implemented(draw)
    cases = [
        ("lognormal", {"mean_log": 0.0, "sigma": 0.1}),
        ("normal", {"mean": 0.0, "sigma": 1.0}),
        ("bernoulli", {"p": 0.5}),
        ("categorical", {"probs": [0.5, 0.5]}),
    ]
    for dist, params in cases:
        for shape in [(), (5,), (3, 4)]:
            got = np.asarray(draw(rng_seed, "yield", 0, 0, shape=shape, dist=dist, **params))
            assert got.shape == shape, (dist, shape)


@pytest.mark.skeleton
def test_draw_rejects_an_unknown_purpose_or_distribution(rng_seed, implemented) -> None:
    """An unrecognised `purpose` or `dist` raises rather than silently aliasing a stream.

    Assertion: `draw` raises `ValueError` when `purpose` is not in `gosplan.rng.PURPOSES` and when
    `dist` is not one of `("lognormal", "normal", "bernoulli", "categorical")`; the message names
    the offending value. Reason (`PURPOSES` docstring): an unrecognised purpose would hash to some
    key and quietly share a stream with another mechanism, which is exactly the class of bug
    key-based randomness exists to make impossible.
    """
    from gosplan.rng import draw

    implemented(draw)
    with pytest.raises(ValueError) as bad_purpose:
        draw(rng_seed, "not_a_purpose", 0, shape=(2,), dist="normal", mean=0.0, sigma=1.0)
    assert "not_a_purpose" in str(bad_purpose.value)
    with pytest.raises(ValueError) as bad_dist:
        draw(rng_seed, "yield", 0, shape=(2,), dist="not_a_dist")
    assert "not_a_dist" in str(bad_dist.value)


@pytest.mark.skeleton
def test_no_module_level_generator_and_no_direct_numpy_random_in_env() -> None:
    """CONTRACT rule 9, checked statically over the source tree.

    Assertion: no module in `gosplan/env/` contains a call to `numpy.random` or `jax.random` (an
    import of `numpy` for array maths is fine; a call into its random namespace is not), and no
    module in `gosplan/` binds a `Generator` or `RandomState` at module scope. The check reads the
    source of every module in the package - by AST, so that a mention inside a docstring is not a
    false positive.

    This is the static half of CONTRACT rule 9; the behavioural half is the determinism and
    order-independence of T-U6 above. `seed_policy` is a separate stream and agents draw from a
    `numpy.random.Generator` they are handed, never from a global.
    """
    root = pathlib.Path(__file__).resolve().parents[2] / "gosplan"
    env_dir = root / "env"

    random_calls: list[str] = []
    module_generators: list[str] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            # a call into numpy.random / jax.random, anywhere under gosplan/env/
            if isinstance(node, ast.Attribute) and env_dir in path.parents:
                parts = []
                cur: ast.AST = node
                while isinstance(cur, ast.Attribute):
                    parts.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    parts.append(cur.id)
                dotted = ".".join(reversed(parts))
                if dotted.startswith(("numpy.random", "np.random", "jax.random")):
                    random_calls.append(f"{path.name}:{node.lineno} {dotted}")
        # a Generator / RandomState bound at module scope, anywhere under gosplan/
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            src = ast.unparse(node)
            if "default_rng(" in src or "Generator(" in src or "RandomState(" in src:
                module_generators.append(f"{path.name}:{node.lineno} {src[:60]}")

    assert random_calls == [], f"CONTRACT rule 9: direct RNG use in gosplan/env/: {random_calls}"
    assert module_generators == [], f"CONTRACT rule 9: module-level generator: {module_generators}"
