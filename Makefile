# gosplan-env — plan-fulfilment MARL environment. Requires uv and GNU make.
UV ?= uv

.PHONY: setup test spec-check golden lint format gate clean

setup:
	$(UV) sync
	-$(UV) run pre-commit install

test:
	$(UV) run pytest -q

# PLAN section 12.3 WO-001: spec/spec.py imports and every public symbol of
# PLAN section 10 is present.
spec-check:
	$(UV) run pytest tests/unit/test_spec_imports.py -q

# PLAN section 11: the golden matrix is generated from ref/, never hand-written, so
# numeric expectations are never hand-computed into test files (finding F14).
# tests/golden/*.json is git-ignored; a fresh checkout regenerates it here.
# Regenerated only by the lead, and only with a spec/CHANGELOG.md entry (WO-013).
golden:
	$(UV) run python -m ref.gen_golden

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format:
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

# CONTRACT rule 13: gates are experiments, not tests. This target refuses to run them.
gate:
	@echo "Refusing to run gates automatically."
	@echo "Gates G0-G4 (PLAN section 13) are lead-run experiments held in tests/acceptance/."
	@echo "Nothing there is a unit test, nothing there is on a work order's must-pass list,"
	@echo "and no implementer session runs it (CONTRACT rule 13). It is excluded from"
	@echo "pytest testpaths, so 'make test' and CI never collect it."
	@echo "A gate is run deliberately by the lead, and its artefacts (PLAN section 13) are"
	@echo "written under runs/ with the manifest of CONTRACT rule 10."
	@exit 1

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .hypothesis
