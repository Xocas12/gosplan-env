AMBIGUITY REPORT   WO-020 (criterion 2)   raised by the LEAD during the labelled study

## Issue

AMBIGUITY-011 resolution 3 states the notched-arm condition as: learned share of REPORT rows in
[1.00, 1.02] >= 0.440 AND the seed-level CI of `b_hat` excludes 0. The first notched-arm runs of
the labelled study (attempt-2 learner, `N = 20`, `a*pen` = 20) put 99.8-100% of their measured
reports inside the window. With (almost) no mass outside the excluded window [0.95, 1.02], the
polynomial counterfactual has no support: `b_hat` is +inf or a very large number, and the
bootstrap CI is NaN in some seeds. The pre-registration never anticipated a learned policy this
concentrated; "CI excludes 0" is then undefined, and reading NaN as "not met" or as "met" is a
choice made after seeing data.

## Ruling (LEAD, recorded before the study finished)

Neither reading is chosen. Each seed's criterion-2 status is `pass`, `fail` or `undefined`
(finite-CI conditions as pre-registered; `undefined` when the CI is not finite and the share
condition holds, for the notched arm, or whenever the CI is not finite, for the smooth arm). The
report states the pass fraction and the criterion-2 verdict under both readings - undefined as
not met (strict) and undefined as met - and the number of undefined seeds. The run is a labelled
study (AMBIGUITY-021), so no gate verdict rests on the choice. The report is regenerated from the
cached per-run measurements after the study finishes (no retraining), because the running process
loaded the earlier report code.
