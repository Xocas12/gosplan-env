AMBIGUITY REPORT   WO-010 / WO-014   gosplan/agents/heuristic.py (DPGreedy)
Question (one sentence):
How does `DPGreedy` map an enterprise's observed `(T, S)` onto the DP's log-spaced target grid and
linear stock grid, and which productivity `A` does that grid use?

What the spec says / does not say (quote):
The WO-010 card and the `DPGreedy` docstring say "nearest grid point, no interpolation"; the target
axis is log-spaced on `[T_min, target_hi_mult * A * cap]` and the stock axis linear on `[0, S_max]`.
Neither says whether "nearest" is measured in `T` or `log T`, how ties and off-grid values are
handled, or which `A` builds the grid when sectors differ (`DPSolution` holds one grid).

Options considered: nearest in linear `T` (arithmetic cell midpoints) vs nearest in `log T`
(geometric midpoints); per-enterprise grids vs one grid.

Impact: the two metrics agree only on grid points; the WO-014 must-pass item ("DPGreedy reproduces
the DP policy inside the env at N = 1") depends on the choice.

Tests blocked: tests/unit/test_dp.py (the DPGreedy item).

LEAD RESOLUTION (2026-09-24):
Target axis: nearest in `log T` (the axis is log-spaced, so its natural metric is logarithmic).
Stock axis: nearest in linear `S`. Ties go to the lower index; values beyond either end clamp to
the end point (no extrapolation). The grid uses the enterprise's own sector productivity; when
productivities differ across sectors `DPGreedy` raises `ValueError`, since one `DPSolution` carries
one grid (all Phase-1 productivities are 1.0).
