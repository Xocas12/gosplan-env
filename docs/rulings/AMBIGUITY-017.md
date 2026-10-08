AMBIGUITY REPORT   WO-016   gosplan/metrics/phenomena.py
A. Bootstrap unit inside one ledger (one training seed): the ledger has no seed column.
B. How the per-period `padding_index = val_measured / val_true` aggregates over the window.
C. How `phenomenon_padding(ledger, cfg)` - one ledger, one `a*pen` level - reports an elasticity
   "across the three levels".

LEAD RESOLUTION (2026-09-24):
A. Resample EVALUATION EPISODES. Under AMBIGUITY-016 every evaluation episode has its own
   `seed_env`, so episodes are independent draws of the environment for one trained policy - the
   within-seed variability a per-seed CI (G2 criterion 2, "CI excluding 0 in >= 90% of 30 seeds")
   needs. Harnesses must therefore evaluate many episodes per trained seed (WO-020 uses the
   `TrainConfig.eval_episodes` it records); a one-episode ledger yields a degenerate CI and is
   reported as such.
B. Ratio of window means over one value per (episode, period):
   `mean_t val_measured_t / mean_t val_true_t`; NaN when the mean `val_true` is 0.
C. `phenomenon_padding` returns the per-level `padding` (and `padding_index`); criterion 3
   ("monotone decreasing and within 0.03 of the DP at each level") is evaluated by WO-020 from the
   three per-level calls, so no elasticity formula is needed or invented.
