"""Key-based randomness: the single source of every environment draw.

Realises: PLAN section 2.15 (RNG, key-based rather than stream-based), with the common-random-number
requirement of PLAN section 4.3 and the NumPy/JAX parity requirement of PLAN section 12.4 (WO-029).
Owning work order: **WO-004** (RNG; P1; MID-fast; depends on WO-001). Must pass
`tests/unit/test_rng.py` (test **T-U6**), which covers the four distributions
`lognormal(mean_log, sigma)`, `normal`, `bernoulli` and `categorical`.

CONTRACT rule 9, verbatim:

    9. RNG. All environment randomness goes through rng.draw(seed_env, purpose, *indices).
       No direct calls to numpy.random / jax.random in gosplan/env/. seed_policy is separate.

Design (PLAN section 2.15). A draw is identified by a **key**, not by a position in a stream:
`numpy.random.SeedSequence([seed_env, crc32(purpose), *indices])` spawns an independent generator
for that key, and the values are read from it. `purpose` names what is being drawn (the tuple
`PURPOSES` below is the closed list) and `*indices` are the integer coordinates of the draw,
conventionally `t`, then `k`, then `i`, with the trailing index vectorised through `shape`.

Three consequences the rest of the design leans on, and which the implementation must not break:

  * **Order independence.** The value of a draw depends only on its key, never on how many draws
    were taken before it. So the NumPy environment and the Phase-2 JAX port (WO-029) agree by
    construction, and reordering the period schedule of PLAN section 2.5 cannot silently change a
    trajectory.
  * **Common random numbers.** Two arms that share `seed_env` see identical shocks for identical
    keys, which is what makes the paired contrasts of PLAN section 4.3 and the CRN comparisons of
    PLAN section 4.1 (phenomenon 2 against the truthful-myopic baseline under the same
    delivery-timing draws) valid without variance-reduction tricks.
  * **Stream separation.** `seed_policy` (`TechConfig.seed_policy`) drives policy sampling and
    exploration and never enters a key here; `seed_env` never seeds a policy. Mixing them would
    make an environment shock depend on the agent, destroying CRN.

**No global generators.** This module holds no module-level `Generator`, no cached `SeedSequence`,
no counter and no memo keyed on anything but the arguments of a call: a global would reintroduce
call-order dependence through the back door and is forbidden by the WO-004 card. `draw` is a pure
function of `(seed_env, purpose, indices, shape, dist, params)`. If the cost of respawning a
generator per key ever matters, the fix is a pure, key-addressed cache proposed through an
AMBIGUITY REPORT (CONTRACT rule 3), not a hidden stream.

**Skeleton status.** The purpose list and the type aliases are real content; `draw` raises
`NotImplementedError` until WO-004 lands.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

Array = np.ndarray
"""Alias for every numeric array in the interface (PLAN section 10), mirroring `spec.spec.Array`.
The Phase-2 JAX port substitutes its own array type behind the same name, so no signature here may
rely on numpy-only methods."""

Purpose = Literal[
    "yield",
    "audit",
    "auditnoise",
    "arrival",
    "channel",
    "drift",
    "terminate",
    "trade_visibility",
    "selfobs",
]
"""The enumerated RNG purposes of PLAN section 2.15, plus `selfobs` for the observation noise of
WO-008. Mirrors `spec.spec.Purpose`. Keying by purpose is what makes draws order-independent."""

Dist = Literal["lognormal", "normal", "bernoulli", "categorical"]
"""The distributions `draw` must support (the WO-004 must-pass list). Mirrors `spec.spec.Dist`.
`lognormal` is parameterised by `(mean_log, sigma)`; the yield shock of PLAN section 2.6 uses
`mean_log = -sigma**2 / 2` so that `E[eps] = 1`."""

PURPOSES: tuple[Purpose, ...] = (
    "yield",
    "audit",
    "auditnoise",
    "arrival",
    "channel",
    "drift",
    "terminate",
    "trade_visibility",
    "selfobs",
)
"""The closed list of legal purposes, as data, in the order PLAN section 2.15 gives them:

    yield             per-step multiplicative yield shock eps_ik (PLAN section 2.6)
    audit             audit selection at the AUDIT step (PLAN section 2.7.4)
    auditnoise        audit measurement error nu in S_hat = S * exp(nu) (PLAN section 2.8)
    arrival           arrival step of a delivered unit when delivery_timing != "uniform"
                      (PLAN section 2.6)
    channel           reporting-channel distortion xi in claimed <- claimed * exp(xi)
                      (PLAN section 2.7.5)
    drift             per-period I-O drift zeta in a <- a * exp(zeta) (PLAN section 2.6)
    terminate         geometric episode termination with continuation probability `tenure`
                      (PLAN section 2.12)
    trade_visibility  which counterparties are visible for bilateral trade (PLAN section 2.13)
    selfobs           noise on the agent's own observation fields, gated by `self_obs_noise`
                      (PLAN section 2.4; added for WO-008)

Two draws with different purposes are independent by construction even at identical indices, which
is why a new kind of randomness is added by adding a purpose here (and to `Purpose` and to
`spec.spec.Purpose`, which needs a `spec/CHANGELOG.md` entry, CONTRACT rule 1) rather than by
reusing an existing one at shifted indices. `draw` rejects a `purpose` outside this tuple: an
unrecognised purpose would silently alias another mechanism's stream.

Binds: `tests/unit/test_rng.py` (test **T-U6**), which checks the tuple against `Purpose` and
checks that distinct purposes give independent draws at the same indices."""


def draw(
    seed_env: int,
    purpose: Purpose,
    *indices: int,
    shape: tuple[int, ...],
    dist: Dist,
    **params: float,
) -> Array:
    """Key-based random draw: the single source of environment randomness (PLAN section 2.15).

    Takes: `seed_env`, the run's root environment seed (`TechConfig.seed_env`); `purpose`, one of
    `PURPOSES`; `*indices`, the integer coordinates of the draw, conventionally `t`, then `k`, then
    `i`; `shape`, the shape of the array to return, vectorising over the trailing index; `dist`,
    one of the values of `Dist`; `**params`, the distribution parameters -

        lognormal    mean_log, sigma      gen.lognormal(mean=mean_log, sigma=sigma, size=shape)
        normal       mean, sigma          gen.normal(loc=mean, scale=sigma, size=shape)
        bernoulli    p                    gen.random(size=shape) < p
        categorical  probs                gen.choice(len(probs), size=shape, p=probs)

    The right-hand column is the EXACT generator call, not an illustration. `ref/ref_step.py`
    re-derives the same table independently and states that it must match element for element, and
    `tests/golden/` compares the two: `gen.random(size=shape) < p` and
    `gen.binomial(1, p, shape).astype(bool)` draw the same distribution but consume the generator
    differently, so a substitution here diverges every later draw and surfaces at WO-009 as an
    unexplained golden-parity break. Pinned by ambiguity report #52; `ref/` is not on WO-004's
    whitelist, so the table has to live here for the card to be executable at all.

    Returns: an `Array` of exactly the requested `shape` (boolean for `bernoulli`, integer for
    `categorical`, float otherwise).

    Construction (PLAN section 2.15, WO-004 notes), which the implementation must follow exactly
    because every reproducibility property below depends on it:

        SeedSequence([seed_env, crc32(purpose), *indices])   ->   Generator   ->   values

    `crc32` is `zlib.crc32(purpose.encode())`: a stable, platform-independent integer for the
    purpose name, unlike Python's salted `hash`. Nothing else enters the key - not a call counter,
    not the wall clock, not the size of a previous draw.

    Properties the construction buys, all of which `tests/unit/test_rng.py` checks:
      * deterministic in `(seed_env, purpose, indices, shape, dist, params)` and **independent of
        call order**, so the NumPy and JAX implementations agree by construction (the WO-029 parity
        test) and the period schedule of PLAN section 2.5 can be reordered safely;
      * common random numbers across arms whenever `seed_env` is shared (PLAN section 4.3);
      * `seed_policy` is an entirely separate stream and never appears in a key here (CONTRACT rule
        9);
      * distinct purposes are independent at identical indices.

    Constraints. CONTRACT rule 9 forbids any direct `numpy.random` or `jax.random` call inside
    `gosplan/env/` - every shock there is a `draw` call - and forbids module-level global
    generators anywhere, so this function creates its generator from the key on each call and keeps
    no state. It raises `ValueError` for a `purpose` outside `PURPOSES`, for a `dist` outside
    `Dist`, and for missing or unknown distribution parameters, rather than defaulting them.

    Interface note. PLAN section 10 types `purpose` and `dist` as `str`; `spec/spec.py` narrows
    them to the `Purpose` and `Dist` literals, which enumerate exactly the values PLAN section 2.15
    and the WO-004 card name. The narrowing is recorded in `spec/CHANGELOG.md` at the v1 freeze.

    Binds: test **T-U6** in `tests/unit/test_rng.py` - `draw` is deterministic in
    `(seed_env, purpose, indices)` and independent of call order, and each distribution has the
    stated moments (the PLAN section 2.6 yield shock has mean 1 to 1e-3 over 1e5 draws, which
    `tests/unit/test_production.py` also checks). Owning WO: **WO-004**.
    """
    raise NotImplementedError("PLAN section 2.15 - implemented in WO-004")
