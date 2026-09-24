AMBIGUITY REPORT   WO-007   gosplan/env/reward.py   (NEEDS HUMAN - economic judgement)
Question (one sentence):
In the smooth counterfactual arm (w = 0.25, rho_cap = inf), should the overfulfilment term keep its
kink at rho = 1, given that T-U3 requires a continuous derivative there?

What the spec says / does not say (quote):
PLAN section 2.8, the WO-007 card, the `bonus` docstring and `ref/ref_step.py::ref_bonus` all give
`B(rho) = beta * Lambda_w(rho - 1) + s * clip(rho - 1, 0, rho_cap - 1)`, with the card adding that at
rho_cap = inf the second term is `s * max(rho - 1, 0)`. That term has a derivative jump of `s` at
rho = 1 for any w. PLAN section 2.8 also says the smooth counterfactual has "no discontinuity and
no kink anywhere", and T-U3 (`test_bonus_has_a_continuous_derivative_only_in_the_smooth_
configuration`) asserts successive central differences differ by < 1e-3; at the P1 slope s = 0.5
the observed jump is 0.25.

Options considered (A/B/…), and why the spec does not decide:
A. Keep the formula verbatim (current code, matches ref and the golden files); T-U3's third clause
   is wrong and is amended.
B. Smooth the slope term in the smooth arm, e.g. `s * w * softplus((rho - 1) / w)`; a new formula,
   requires changing ref/ref_bonus and regenerating the golden set.
C. Set s = 0 in the smooth arm's named configuration.
The formula and the named-arm property contradict each other.

Impact if the wrong option is picked:
The smooth arm is the G2 criterion-2 counterfactual (bunching under a notch but not under a smooth
bonus). A kink at rho = 1 can itself attract mass to rho = 1 and weaken that contrast.

Tests blocked:
tests/unit/test_reward.py::test_bonus_has_a_continuous_derivative_only_in_the_smooth_configuration
(fails; CI `test` is red until resolved).

RESOLUTION (2026-09-24; the human delegated the call to the LEAD's judgement):
B. In the smooth arm only (w > 0 and rho_cap = inf) the slope term is `s * w * softplus((rho - 1) /
w)`, the smooth counterpart of `s * max(rho - 1, 0)` at the notch's own width; the notched (w = 0)
and kink-only (finite rho_cap) arms keep the written formula. Reason: PLAN section 2.8 states the
smooth arm has no kink anywhere, and a kink at rho = 1 would itself pull reports to rho = 1 and
blur the G2 criterion-2 contrast. `ref/ref_step.py::ref_bonus` changed identically; the golden
set was regenerated (the six `smooth` cells changed). Separately, T-U3's `< 1e-3` bound was below
the logistic notch's own curvature at w = 0.25 on the test's 1e-3 grid (about 1.5e-3), so no
formula with the specified notch could pass it; a lead edit loosens it to `< 1e-2`, which a kink
(step about s/2 = 0.25) still fails. PLAN section 2.8's formula text is to be amended at WO-013.
