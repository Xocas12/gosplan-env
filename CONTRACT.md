CONTRACT — gosplan-env
These rules bind every session, human or model. Violations invalidate the session's output.

1. FROZEN SPEC. spec/spec.py is provisional (v0) until gate G1 and frozen (v1) thereafter.
   After v1, only the lead may change it, and only with a spec/CHANGELOG.md entry
   (version, reason, affected work orders). No other session edits spec/spec.py.

2. FROZEN TESTS. tests/unit, tests/behavioural and tests/golden are read-only for
   implementers. If a test looks wrong, file an AMBIGUITY REPORT; do not edit it, do not
   skip it, do not special-case the implementation to pass it.

3. STOP AND REPORT. When the spec, the work order and the whitelisted files do not
   determine a choice, emit an AMBIGUITY REPORT (workorders/AMBIGUITY_TEMPLATE.md) and end
   the session. Fluent invention is the failure mode this rule exists to prevent.
   Choosing "the reasonable default" is a violation.

4. REWARD TERMS. The enterprise reward is exactly:
   production step:  −scale · c_ik
   report step:       scale · (B(ρ) − 1[audited]·Pen + trade_surplus)
   No per-step shaping, no auxiliary reward, no curiosity term, no potential-based term.
   No running reward normalisation (running statistics change the effective reward over
   training and, with heavy-tailed penalties, shrink the notch in normalised units).
   Per-batch advantage normalisation inside PPO is permitted. scale = reward_scale(cfg),
   computed analytically from the configuration.

5. PLANNER BLINDNESS. Planner rules take a PlannerView and nothing else. PlannerView is
   built by make_planner_view() and contains no true quantity. Any planner function whose
   signature accepts State is a violation.

6. WELFARE BLINDNESS. welfare_true and val_measured are logged and never appear in any
   observation, reward, or agent input. The PPO adapter's forward pass takes obs only.

7. NO HARD-CODED PATHOLOGY. No transition rule or reward term may implement bunching,
   padding, storming, hoarding, shaving or trade directly. tests/behavioural/
   test_no_hardcoded_pathology.py checks this behaviourally with heuristic agents; passing
   it is necessary, not sufficient — the lead reviews every env/ diff against this rule.

8. BOUNDS ARE RESULTS. report_ratio is bounded at ρ_max = 10. The fraction of reports at
   the bound is logged; > 1% flags the run manifest BOUND_BINDING and the result is
   reported with the flag. Never silently widen or narrow a bound to fix a result.

9. RNG. All environment randomness goes through rng.draw(seed_env, purpose, *indices).
   No direct calls to numpy.random / jax.random in gosplan/env/. seed_policy is separate.

10. MANIFEST. Every run writes runs/<hash>/manifest.json: config hash, spec version, git
    hash, seeds, reference-PPO version, estimator version, LLM model ids and versions,
    solver version and optimality gap, flags.

11. PARAMETER ARMS. The INFO/INC/SUPPLY/TECH classification in gosplan/params.py is a
    design decision. Changing an arm assignment requires a CHANGELOG entry.

12. WORK ORDERS. An implementer reads only the files on the work order's whitelist,
    writes only the files it names, runs the completion command verbatim, and reports
    in the format the card specifies.

13. TESTS ARE NOT EXPERIMENTS. tests/acceptance/ holds lead-run experiments (gates).
    Nothing there is a unit test, nothing there is on any work order's must-pass list,
    and no implementer session runs it.
