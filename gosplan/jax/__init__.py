"""JAX port of the environment - **Phase 2, LEAD-owned, not yet written** (PLAN section 12.4).

Realises: nothing yet. This package is the `gosplan/jax/` row of the PLAN section 8 layout
("LEAD, P2: port + parity"). Owning work order: **WO-029** (JAX port), one unit, **LEAD**-owned -
PLAN section 1.3, finding F14 names the step function, the JAX port and the PPO adapter as the three
units the lead writes itself rather than delegating. No implementer session writes into this
package.

Why the port exists. Phase 3's Saltelli/Sobol design (PLAN sections 4.3 and 12.5, WO-033) needs
orders of magnitude more environment steps than the NumPy implementation delivers, and the compute
budget of PLAN section 14 assumes the vectorised port. The port is a *translation*, not a second
design: `gosplan/env/` is written struct-of-arrays with leading dimension `N` throughout precisely
so this is mechanical (the conventions note of PLAN section 10).

**Parity requirement (the WO-029 acceptance criterion).** 100 agent-steps driven by
`TruthfulMyopic` must be identical to the NumPy implementation to **1e-5**, under the same key-based
RNG. It is a meaningful test rather than a tautology only because of PLAN section 2.15: every
environment draw is keyed as `SeedSequence([seed_env, crc32(purpose), *indices])` and is therefore
**order-independent**, so the two implementations agree by construction rather than by both walking
the same stream in the same order. A port that reproduced the trajectory only by preserving call
order would be one refactor away from silently diverging. The same property is what gives common
random numbers across arms whenever `seed_env` is shared (PLAN section 4.3), and it is what the
port must not break: no global generator, no stream threading, no `jax.random` key split standing in
for a keyed draw (CONTRACT rule 9).

**Branching.** The environment has data-dependent branches that a traced implementation cannot take
with Python control flow. The three the card names explicitly, all Phase-2 toggles that are off in
Phase 1 and therefore easy to get wrong if they are ported without their own parity cases:

    capital        `Kap_{t+1} = (1 - dep) * Kap_t + matured investment`, with the `pending_invest`
                   buffer of depth `supply.invest_lag` (PLAN sections 2.2, 2.6)
    IRS            the increasing-returns toggle `A_j(Kap) = A_j * (Kap / Kap_0)**alpha_irs`
                   (PLAN section 2.6), which is the identity at `alpha_irs = 0`
    setup cost     the fixed cost `F * 1[e > 0]` in `c = kappa * e**2 + F * 1[e > 0] + kappa_q * q
                   * e` (PLAN section 2.6), a genuine discontinuity at `e = 0`

Each is ported with `lax.cond` or `jnp.where` - `where` where both sides are cheap and defined, and
`lax.cond` where evaluating the untaken side would be wrong or wasteful. The same discipline applies
to the other discontinuities the design deliberately contains and must not smooth away: the notch
`Lambda_w(x) = 1[x >= 0]` at `w = 0` and the `rho_cap = inf` branch of `bonus` (PLAN section 2.8),
the `theta = inf` (`min`) branch of the coverage aggregator (PLAN section 2.6), the `claimed_i == 0`
convention `fill_i = 1` in `deliver` (PLAN section 2.7.3), and the `sigma_c -> 1` Cobb-Douglas limit
of the welfare index (PLAN section 2.9.3). A port that rounds a Heaviside into a sigmoid has changed
the economics, not the backend: `notch_width` is a treatment variable and the smooth arm is a
configuration, never an implementation detail.

**Contract, unchanged by the backend.** `Array` is the alias every signature is written against
(PLAN section 10) and the port substitutes its own array type behind that same name, so no module
may rely on a numpy-only method in a signature. CONTRACT rule 9 governs randomness here exactly as
it does under NumPy. CONTRACT rule 4 governs the reward terms: a vectorised rollout is still
forbidden any per-step shaping and any running reward normalisation. CONTRACT rule 6 governs
`StepInfo`: whatever the port emits for the ledger stays off every agent-facing path.

**Status: skeleton only.** This package intentionally contains no code. It is created now so the
layout of PLAN section 8 is complete and so nothing else has to move when WO-029 is issued after the
Phase-2 spec revision. Until then, `gosplan/env/` is the single implementation of the dynamics, and
the parity test - not this docstring - is what will make the second one trustworthy.
"""
