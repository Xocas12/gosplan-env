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

import pytest


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-004")
def test_draw_is_deterministic_in_seed_purpose_and_indices(rng_seed) -> None:
    """T-U6, first half: the same key always yields the same array.

    Assertion: two calls `draw(seed_env, purpose, *indices, shape=shape, dist=dist, **params)` with
    identical arguments return arrays that are exactly equal (bitwise; `np.array_equal`, tolerance
    0), for every purpose in `gosplan.rng.PURPOSES` and every distribution in `Dist`. Equality must
    hold across separate processes as well as within one, because the golden files of PLAN section
    11 are generated once and compared for the life of the repository.

    Mechanism (WO-004 notes): `numpy.random.SeedSequence([seed_env, crc32(purpose), *indices])`
    spawns an independent generator per key, so the key alone determines the bits.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-004")
def test_draw_is_independent_of_call_order(rng_seed) -> None:
    """T-U6, second half: draws do not depend on how many draws preceded them.

    Assertion: a sequence of draws over several keys taken in one order equals, element by element
    and exactly, the same draws taken in the reverse order and in a shuffled order; interleaving an
    unrelated draw between two calls does not change either result. There is no hidden stream state
    to advance.

    Why it is load-bearing: order-independence is what lets the JAX port of WO-029 vectorise the
    same draws in a different execution order and still match to 1e-9, and what makes common random
    numbers across arms (PLAN section 4.3) a property of the seed rather than of the code path.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-004")
def test_distinct_purposes_are_independent_at_identical_indices(rng_seed) -> None:
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
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-004")
def test_distinct_seeds_and_indices_give_different_draws(rng_seed) -> None:
    """Different keys give different numbers - the complement of determinism.

    Assertion: draws at the same purpose and shape but a different `seed_env`, or a different value
    of any one index (`t`, `k` or `i`), differ; and a draw of shape `(n,)` vectorised over the
    trailing index equals the `n` scalar draws taken at that index one at a time, so the vectorised
    and looped forms of the same mechanism agree exactly (this is the identity `ref/ref_step.py`
    relies on when it reproduces the vectorised environment with a single loop).
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-004")
def test_lognormal_draw_has_the_stated_moments(rng_seed) -> None:
    """`dist="lognormal"` returns `exp(N(mean_log, sigma**2))` with the stated moments.

    Assertion: over a large sample (1e5 draws or more, vectorised over the trailing index), the
    sample mean of `draw(..., dist="lognormal", mean_log=mu, sigma=s)` matches
    `exp(mu + s**2 / 2)` and the sample variance matches
    `(exp(s**2) - 1) * exp(2 * mu + s**2)`, each to 1e-3 in relative terms; every value is strictly
    positive. At the environment's own parameterisation `mean_log = -sigma**2 / 2` the mean is 1,
    which is the property `tests/unit/test_production.py` checks end to end for the yield shock of
    PLAN section 2.6.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-004")
def test_normal_draw_has_the_stated_moments(rng_seed) -> None:
    """`dist="normal"` returns `N(mean, sigma**2)`.

    Assertion: over 1e5 draws the sample mean matches `mean` and the sample standard deviation
    matches `sigma`, each to 1e-2 absolute at `sigma = 1`; the draw is real-valued and finite
    everywhere. This is the distribution behind the audit measurement error
    `S_hat = S * exp(nu)`, `nu ~ N(0, sigma_aud**2)` (PLAN section 2.8), the channel distortion
    `xi ~ N(0, sigma_ch**2)` (PLAN section 2.7.5), the I-O drift `zeta` (PLAN section 2.6) and the
    self-observation noise (PLAN section 2.4).
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-004")
def test_bernoulli_draw_returns_booleans_at_the_stated_rate(rng_seed) -> None:
    """`dist="bernoulli"` returns a boolean array whose mean is `p`.

    Assertion: `draw(..., shape=(n,), dist="bernoulli", p=p)` has boolean dtype and, over 1e5
    draws, an empirical frequency within 3 binomial standard errors of `p`; `p = 0` gives all
    False and `p = 1` all True. This is the audit selection of PLAN section 2.7.4,
    `audited_i ~ Bernoulli(a)` with key `(seed_env, "audit", t, i)`, whose empirical rate
    `tests/unit/test_planner.py` checks against `cfg.information.audit_rate`.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-004")
def test_categorical_draw_returns_indices_with_the_stated_probabilities(rng_seed) -> None:
    """`dist="categorical"` returns integer category indices distributed as `probs`.

    Assertion: `draw(..., shape=(n,), dist="categorical", probs=probs)` returns an integer array
    with every value in `[0, len(probs))`, and over 1e5 draws each category's empirical frequency
    is within 3 standard errors of its probability. This is the within-period arrival step of a
    delivered unit under `delivery_timing != "uniform"` (PLAN section 2.6, purpose `arrival`), whose
    Phase-2 locked value is the uniform `arrival_probs = (0.25, 0.25, 0.25, 0.25)` of PLAN section
    4.2.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-004")
def test_shape_argument_controls_the_returned_shape(rng_seed) -> None:
    """`shape` is the shape of the array returned, for every distribution.

    Assertion: for each `dist`, `draw(..., shape=s, ...).shape == s` for scalar `()`, vector `(n,)`
    and matrix `(n, m)` shapes, with the vectorisation running over the trailing index as PLAN
    section 2.15 specifies. The environment relies on this to draw one `eps_ik` per enterprise in a
    single call keyed `(seed_env, "yield", t, k)` while the reference implementation draws them one
    at a time keyed `(seed_env, "yield", t, k, i)`; the two must agree, which is the identity the
    previous test states.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-004")
def test_draw_rejects_an_unknown_purpose_or_distribution(rng_seed) -> None:
    """An unrecognised `purpose` or `dist` raises rather than silently aliasing a stream.

    Assertion: `draw` raises `ValueError` when `purpose` is not in `gosplan.rng.PURPOSES` and when
    `dist` is not one of `("lognormal", "normal", "bernoulli", "categorical")`; the message names
    the offending value. Reason (`PURPOSES` docstring): an unrecognised purpose would hash to some
    key and quietly share a stream with another mechanism, which is exactly the class of bug
    key-based randomness exists to make impossible.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-004")
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
    assert False
