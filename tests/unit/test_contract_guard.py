"""Tests for `scripts/contract_guard.py` - the static CONTRACT checker.

NOT PART OF THE FROZEN SURFACE. CONTRACT rule 2 makes `tests/unit`, `tests/behavioural` and
`tests/golden` read-only for implementers, because those directories hold the frozen expectations
that the environment implementation is measured against (PLAN section 11). This file is not one of
them: it is the unit test of a CI script, it makes no claim about environment behaviour, it is on
no work order's must-pass list, and it may be edited by whoever edits `scripts/contract_guard.py`
- in the same pull request, or the guard and its test drift apart. The guard therefore carves this
one path out STRUCTURALLY, in `FROZEN_TEST_NON_FROZEN_PATHS`, rather than by a standing entry in
`.github/FROZEN_TEST_EXEMPTION`: that file is for temporary, reviewable suppressions of genuine
lead edits to the frozen suite, and a permanent entry in it would be a permanently disabled rule.
See `scripts/README.md`, "The rule-2 exemption".

THESE ARE REAL, PASSING TESTS, not skeleton stubs. Nothing here carries `@pytest.mark.skeleton` or
`@pytest.mark.skip`, and no body raises `NotImplementedError`: the guard is fully implemented CI
infrastructure, so its tests are too.

METHOD. Each rule gets a tiny synthetic tree under `tmp_path` that violates it, and a clean twin
that does not; the check function is called directly, and the assertion is on the specific rule
firing rather than on any violation appearing. Two tests carry more weight than the rest:

  * `test_rule_4_denylist_in_docstring_only_does_not_fire` - the string/comment distinction is the
    whole point of the AST route. `gosplan/env/reward.py` names `RunningMeanStd` as FORBIDDEN in
    prose, and `gosplan/env/obs.py` holds `"welfare_true"` as a string literal in `NEVER_OBSERVED`;
    a guard that flagged those would be turned off within a day.
  * `test_real_repository_tree_passes_run_all` - the repository as it stands is clean, so a
    violation introduced later is unambiguously the fault of the change that introduced it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARD_PATH = REPO_ROOT / "scripts" / "contract_guard.py"


def _load_guard():
    """Import `scripts/contract_guard.py` by path.

    `scripts/` is deliberately not a package - the guard is a standalone CI script that must run on
    a bare interpreter, before `uv sync`, with no import machinery around it - so it is loaded here
    from its file location rather than imported by name.
    """
    spec = importlib.util.spec_from_file_location("contract_guard", GUARD_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


guard = _load_guard()


# -------------------------------------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------------------------------------


def write(root: Path, rel: str, text: str) -> Path:
    """Create `root/rel` with `text`, making parent directories as needed."""
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def rules(violations) -> list[str]:
    """The rule ids of a violation list, for assertions that name the rule rather than the count."""
    return [v.rule for v in violations]


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A minimal synthetic repository: the directories the checks look at, and nothing else.

    `tmp_path` is not a git work tree, so every git-dependent check degrades to its documented
    fallback. That is deliberate: the fallbacks are part of the contract this script offers CI.
    """
    for rel in ("gosplan/env", "gosplan/agents/ppo", "gosplan/metrics", "gosplan/experiments"):
        (tmp_path / rel).mkdir(parents=True, exist_ok=True)
    write(
        tmp_path,
        "pyproject.toml",
        '[tool.pytest.ini_options]\ntestpaths = ["tests/unit", "tests/behavioural"]\n',
    )
    write(tmp_path, ".gitignore", "PLAN.md\n")
    return tmp_path


# -------------------------------------------------------------------------------------------------
# rule 1 - FROZEN SPEC
# -------------------------------------------------------------------------------------------------


def test_rule_1_spec_change_without_changelog_fires(tree: Path) -> None:
    """Touching `spec/spec.py` without touching `spec/CHANGELOG.md` is a rule-1 violation."""
    write(tree, "spec/spec.py", "SPEC_VERSION = '0.1.0'\n")
    found = guard.check_spec_freeze(tree, {"spec/spec.py", "gosplan/env/planner.py"})
    assert rules(found) == ["1"]
    assert "spec/CHANGELOG.md" in found[0].message
    assert "rule 1" in found[0].message.lower()
    assert found[0].fix


def test_rule_1_is_skipped_without_a_diff(tree: Path) -> None:
    """With `changed=None` there is no diff to judge, so rule 1 reports nothing."""
    write(tree, "spec/spec.py", "SPEC_VERSION = '0.1.0'\n")
    assert guard.check_spec_freeze(tree, None) == []


def test_rule_1_ignores_diffs_that_leave_the_spec_alone(tree: Path) -> None:
    """A diff that does not touch `spec/spec.py` is not a rule-1 matter."""
    assert guard.check_spec_freeze(tree, {"gosplan/env/reward.py", "README.md"}) == []


def test_rule_1_version_headings_are_extracted(tree: Path) -> None:
    """`_changelog_versions` reads version numbers and ignores an 'Unreleased' heading."""
    text = "# CHANGELOG\n\n## Unreleased\n\n## 1.0.0 - 2026-09-05\n\n## 0.1.0 - 2026-09-01\n"
    assert guard._changelog_versions(text) == {"1.0.0", "0.1.0"}


def test_rule_1_baseline_never_falls_back_to_head(tree: Path, monkeypatch) -> None:
    """`_changelog_on_main` consults `main` and `origin/main` and then gives up - never `HEAD`.

    `HEAD` is the branch being judged. Using it as the baseline makes the set difference
    unconditionally empty, which is a guaranteed false rule-1 violation rather than a skip.
    """
    refs: list[str] = []

    def record(root: Path, *args: str) -> tuple[int, str]:
        refs.append(args[1].split(":", 1)[0])
        return 1, ""

    monkeypatch.setattr(guard, "_git", record)
    assert guard._changelog_on_main(tree) is None
    assert refs == ["main", "origin/main"]


def test_rule_1_without_a_main_ref_skips_rather_than_firing(
    tree: Path, monkeypatch, capsys
) -> None:
    """A correct spec revision on a checkout with no `main` is a notice, not a violation.

    Reproduces the detached-HEAD / single-branch case: `main` and `origin/main` are unresolvable
    while `HEAD` resolves fine. The changelog carries a genuinely new version heading, so the only
    correct outcome is to skip the new-heading half with the notice - the old `HEAD` fallback made
    the baseline equal to the file under test and reported rule 1 against a clean change.
    """
    changelog = "# CHANGELOG\n\n## 0.2.0 - 2026-02-01\n\nSpec v1 freeze at gate G1 (WO-013).\n"
    write(tree, "spec/spec.py", "SPEC_VERSION = '0.2.0'\n")
    write(tree, "spec/CHANGELOG.md", changelog)

    def only_head_resolves(root: Path, *args: str) -> tuple[int, str]:
        if args[0] == "show" and args[1].startswith("HEAD:"):
            return 0, changelog
        return 1, ""

    monkeypatch.setattr(guard, "_git", only_head_resolves)
    assert guard.check_spec_freeze(tree, {"spec/spec.py", "spec/CHANGELOG.md"}) == []
    assert "the rule-1 new-version-heading half is skipped" in capsys.readouterr().err


# -------------------------------------------------------------------------------------------------
# rule 2 - FROZEN TESTS
# -------------------------------------------------------------------------------------------------


def test_rule_2_frozen_test_edit_is_review_required(tree: Path, monkeypatch) -> None:
    """A frozen-test file in the diff is reported at severity `review-required`."""
    monkeypatch.delenv(guard.FROZEN_TEST_ENV_VAR, raising=False)
    found = guard.check_frozen_tests(tree, {"tests/behavioural/test_planner_blindness.py"})
    assert rules(found) == ["2"]
    assert found[0].severity == "review-required"
    assert "AMBIGUITY REPORT" in found[0].fix
    assert guard.FROZEN_TEST_EXEMPTION_PATH in found[0].fix


def test_rule_2_covers_all_three_frozen_directories(tree: Path, monkeypatch) -> None:
    """`tests/unit`, `tests/behavioural` and `tests/golden` are all frozen; nothing else is."""
    monkeypatch.delenv(guard.FROZEN_TEST_ENV_VAR, raising=False)
    changed = {
        "tests/unit/test_reward.py",
        "tests/behavioural/test_fixed_point.py",
        "tests/golden/test_golden_parity.py",
        "tests/acceptance/gate_g2_phase1.py",
        "tests/conftest.py",
        "gosplan/env/reward.py",
    }
    found = guard.check_frozen_tests(tree, changed)
    assert {v.path for v in found} == {
        "tests/unit/test_reward.py",
        "tests/behavioural/test_fixed_point.py",
        "tests/golden/test_golden_parity.py",
    }


def test_rule_2_exemption_file_suppresses_named_paths(tree: Path, monkeypatch) -> None:
    """`.github/FROZEN_TEST_EXEMPTION` suppresses exactly the paths and prefixes it names."""
    monkeypatch.delenv(guard.FROZEN_TEST_ENV_VAR, raising=False)
    write(
        tree,
        guard.FROZEN_TEST_EXEMPTION_PATH,
        "# lead is editing the frozen suite for the v1 freeze (WO-013)\n"
        "tests/unit/test_spec_imports.py\n"
        "tests/golden/\n",
    )
    changed = {
        "tests/unit/test_spec_imports.py",
        "tests/golden/test_golden_parity.py",
        "tests/unit/test_reward.py",
    }
    found = guard.check_frozen_tests(tree, changed)
    assert [v.path for v in found] == ["tests/unit/test_reward.py"]


def test_rule_2_the_guards_own_test_is_carved_out_structurally(tree: Path, monkeypatch) -> None:
    """THE PULL REQUEST THAT ADDS THIS GUARD MUST PASS IT.

    `tests/unit/test_contract_guard.py` is a permanent resident of a frozen directory that is not
    part of the frozen surface, so it is skipped by name with NO exemption file present and no
    environment variable set. Without this, the contract job fails on the very change that
    introduces the guard, and the only remedies are a standing exemption entry - which the README
    itself calls a disabled rule - or switching rule 2 off in a workflow.
    """
    monkeypatch.delenv(guard.FROZEN_TEST_ENV_VAR, raising=False)
    assert not (tree / guard.FROZEN_TEST_EXEMPTION_PATH).exists()
    changed = {
        "scripts/contract_guard.py",
        "scripts/README.md",
        "tests/unit/test_contract_guard.py",
        "tests/unit/test_reward.py",
    }
    found = guard.check_frozen_tests(tree, changed)
    assert [v.path for v in found] == ["tests/unit/test_reward.py"]


def test_rule_2_carve_out_names_only_the_guards_own_test() -> None:
    """The carve-out stays this short: adding a path unfreezes that file for good."""
    assert guard.FROZEN_TEST_NON_FROZEN_PATHS == frozenset({"tests/unit/test_contract_guard.py"})


def test_rule_2_environment_variable_suppresses_everything(tree: Path, monkeypatch) -> None:
    """`CONTRACT_ALLOW_FROZEN_TESTS=1` turns rule 2 off for the run, as documented."""
    monkeypatch.setenv(guard.FROZEN_TEST_ENV_VAR, "1")
    assert guard.check_frozen_tests(tree, {"tests/unit/test_reward.py"}) == []


def test_rule_2_is_skipped_without_a_diff(tree: Path, monkeypatch) -> None:
    """Rule 2 is a property of a change, so it reports nothing on a whole-tree run."""
    monkeypatch.delenv(guard.FROZEN_TEST_ENV_VAR, raising=False)
    assert guard.check_frozen_tests(tree, None) == []


# -------------------------------------------------------------------------------------------------
# rule 4 - REWARD TERMS
# -------------------------------------------------------------------------------------------------


def test_rule_4_running_normalisation_in_reward_fires(tree: Path) -> None:
    """A `RunningMeanStd` wrapper constructed in `gosplan/env/reward.py` is a rule-4 violation."""
    write(
        tree,
        "gosplan/env/reward.py",
        "from stable_baselines3.common.running_mean_std import RunningMeanStd\n"
        "\n"
        "def enterprise_reward(state, cfg):\n"
        "    normaliser = RunningMeanStd()\n"
        "    return normaliser\n",
    )
    found = guard.check_reward_terms(tree, None)
    assert rules(found) == ["4", "4"]  # the import alias and the call site
    assert all("RunningMeanStd" in v.message for v in found)


def test_rule_4_covers_the_ppo_package(tree: Path) -> None:
    """The denylist applies under `gosplan/agents/ppo/` as well as in the reward module."""
    write(
        tree,
        "gosplan/agents/ppo/adapter.py",
        "def build(env, cfg):\n    return wrap(env, reward_norm=True)\n",
    )
    found = guard.check_reward_terms(tree, None)
    assert rules(found) == ["4"]
    assert found[0].path == "gosplan/agents/ppo/adapter.py"


def test_rule_4_denylist_in_docstring_only_does_not_fire(tree: Path) -> None:
    """THE LOAD-BEARING TEST: prose naming a forbidden construct is not a violation.

    `gosplan/env/reward.py` reproduces CONTRACT rule 4 in full and states that "a `RunningMeanStd`
    wrapper on rewards is a rule-4 violation"; `gosplan/agents/ppo/adapter.py` requires "no
    `NormalizeReward` wrapper". Docstrings, ordinary string literals and comments must all be
    exempt, or the guard reports its own documentation and gets switched off.
    """
    write(
        tree,
        "gosplan/env/reward.py",
        '"""Reward terms (CONTRACT rule 4).\n'
        "\n"
        "FORBIDDEN, and named here so the prohibition is visible at the point of temptation:\n"
        "RunningMeanStd, NormalizeReward, VecNormalize, reward_norm, running_reward, curiosity,\n"
        "intrinsic_reward, potential_based, shaping, bonus_shaping, reward_scale_running.\n"
        'No per-step shaping, no auxiliary reward, no curiosity term."""\n'
        "\n"
        "# A comment may also say VecNormalize and intrinsic_reward without being code.\n"
        "FORBIDDEN_NAMES = ('RunningMeanStd', 'NormalizeReward', 'reward_norm', 'shaping')\n"
        "\n"
        "def enterprise_reward(state, cfg):\n"
        '    """No shaping, no curiosity, no potential_based term."""\n'
        "    return scale * (bonus(rho, cfg) - penalty + trade_surplus)\n",
    )
    assert guard.check_reward_terms(tree, None) == []


def test_rule_4_matches_whole_identifiers_only(tree: Path) -> None:
    """`reward_normalisation` and `reward_scale` are not `reward_norm` or `reward_scale_running`."""
    write(
        tree,
        "gosplan/env/reward.py",
        "def reward_scale(cfg):\n"
        "    reward_normalisation = False\n"
        "    return 1.0 / bonus(1.1, cfg) if not reward_normalisation else 0.0\n",
    )
    assert guard.check_reward_terms(tree, None) == []


# -------------------------------------------------------------------------------------------------
# rule 5 - PLANNER BLINDNESS
# -------------------------------------------------------------------------------------------------


PLANNER_HEADER = (
    '"""Planner rules. CONTRACT rule 5: any planner function taking a State is a violation."""\n'
    "\n"
    "from __future__ import annotations\n"
    "\n"
    "from gosplan.env.state import State\n"
    "\n"
)


def test_rule_5_non_whitelisted_function_taking_state_fires(tree: Path) -> None:
    """A planner rule whose signature names `State` is a rule-5 violation."""
    write(
        tree,
        "gosplan/env/planner.py",
        PLANNER_HEADER + "def make_planner_view(state: State, cfg) -> 'PlannerView':\n    ...\n"
        "\n"
        "def update_targets(state: State, cfg):\n    ...\n",
    )
    found = guard.check_planner_blindness(tree, None)
    assert rules(found) == ["5"]
    assert "update_targets" in found[0].message
    assert "make_planner_view" in found[0].message


@pytest.mark.parametrize(
    "annotation",
    ["State", '"State"', "State | None", "Optional[State]", "list[State]", "spec.State"],
)
def test_rule_5_sees_state_however_it_is_written(tree: Path, annotation: str) -> None:
    """Wrapped, qualified, unioned and stringified annotations all resolve to `State`."""
    write(
        tree,
        "gosplan/env/planner.py",
        PLANNER_HEADER + f"def allocate(view, cfg, extra: {annotation}):\n    ...\n",
    )
    found = guard.check_planner_blindness(tree, None)
    assert rules(found) == ["5"], f"{annotation} was not recognised as State"


def test_rule_5_whitelisted_functions_do_not_fire(tree: Path) -> None:
    """`make_planner_view` and the documented `deliver` may take a `State`; nothing else may."""
    write(
        tree,
        "gosplan/env/planner.py",
        PLANNER_HEADER + "def make_planner_view(state: State, cfg) -> 'PlannerView':\n    ...\n"
        "\n"
        "def deliver(state: State, alloc, cfg):\n    ...\n"
        "\n"
        "def update_targets(view: 'PlannerView', cfg):\n    ...\n"
        "\n"
        "def select_audits(view: 'PlannerView', cfg, t: int):\n    ...\n",
    )
    assert guard.check_planner_blindness(tree, None) == []


def test_rule_5_whitelist_matches_the_frozen_test(tree: Path) -> None:
    """The guard's whitelist and test T-B4's `STATE_ARGUMENT_WHITELIST` must not drift apart."""
    write(tree, "gosplan/env/planner.py", PLANNER_HEADER)
    write(
        tree,
        guard.PLANNER_BLINDNESS_TEST_PATH,
        'STATE_ARGUMENT_WHITELIST = ("make_planner_view", "deliver", "allocate")\n',
    )
    found = guard.check_planner_blindness(tree, None)
    assert rules(found) == ["5"]
    assert "allocate" in found[0].message

    write(
        tree,
        guard.PLANNER_BLINDNESS_TEST_PATH,
        'STATE_ARGUMENT_WHITELIST = ("make_planner_view", "deliver")\n',
    )
    assert guard.check_planner_blindness(tree, None) == []


def test_rule_5_guard_and_repository_frozen_test_agree() -> None:
    """The same drift check, run against the real frozen test rather than a synthetic one."""
    test_file = REPO_ROOT / guard.PLANNER_BLINDNESS_TEST_PATH
    if not test_file.is_file():  # pragma: no cover - the frozen test is committed
        pytest.skip(f"{guard.PLANNER_BLINDNESS_TEST_PATH} is absent")
    import ast

    declared = guard._module_literal(
        ast.parse(test_file.read_text(encoding="utf-8")), "STATE_ARGUMENT_WHITELIST"
    )
    assert declared == guard.PLANNER_STATE_WHITELIST == ("make_planner_view", "deliver")


# -------------------------------------------------------------------------------------------------
# rule 6 - WELFARE BLINDNESS
# -------------------------------------------------------------------------------------------------


def test_rule_6_welfare_reference_in_obs_fires(tree: Path) -> None:
    """Any executable reference to a logged-only quantity inside `obs.py` is a rule-6 violation."""
    write(
        tree,
        "gosplan/env/obs.py",
        "def build_observation(state, cfg, deliv, need):\n"
        "    welfare = welfare_true(state.consumer, cfg)\n"
        "    return [welfare]\n",
    )
    found = guard.check_welfare_blindness(tree, None)
    assert rules(found) and set(rules(found)) == {"6"}
    assert any("welfare_true" in v.message for v in found)


def test_rule_6_welfare_reference_in_an_agent_entry_point_fires(tree: Path) -> None:
    """`act`, `forward`, `observe` and `build_observation` are the four agent entry points."""
    write(
        tree,
        "gosplan/agents/heuristic.py",
        "class Padder:\n"
        "    def act(self, obs, phase, rng):\n"
        "        return self.policy(obs, val_measured)\n",
    )
    found = guard.check_welfare_blindness(tree, None)
    assert rules(found) == ["6"]
    assert "act()" in found[0].message
    assert found[0].path == "gosplan/agents/heuristic.py"


def test_rule_6_string_literals_in_never_observed_do_not_fire(tree: Path) -> None:
    """THE OTHER LOAD-BEARING EXEMPTION: `obs.py` names these quantities as data, on purpose.

    `NEVER_OBSERVED` holds them as string literals so that test T-B5 can iterate the list and plant
    a sentinel in each. That is the rule being enforced, not broken.
    """
    write(
        tree,
        "gosplan/env/obs.py",
        '"""Observation assembly. CONTRACT rule 6: welfare_true and val_measured are logged and\n'
        'never observed."""\n'
        "\n"
        "NEVER_OBSERVED = (\n"
        '    "welfare_true",\n'
        '    "val_measured",\n'
        '    "val_true",\n'
        '    "welfare",\n'
        ")\n"
        "\n"
        "def build_observation(state, cfg, deliv, need):\n"
        '    """Never writes welfare_true or val_measured into obs (rule 6)."""\n'
        "    return state.cum_output / state.target\n",
    )
    assert guard.check_welfare_blindness(tree, None) == []


def test_rule_6_ignores_functions_that_are_not_agent_entry_points(tree: Path) -> None:
    """The metrics layer computes these quantities; that is what they are for."""
    write(
        tree,
        "gosplan/metrics/ledger.py",
        "def write_row(state, cfg):\n    return {'val_true': val_true(state, cfg)}\n",
    )
    assert guard.check_welfare_blindness(tree, None) == []


def test_rule_6_welfare_in_the_reward_module_fires(tree: Path) -> None:
    """THE REWARD CLAUSE: rule 6 forbids these quantities in a REWARD as well as an observation.

    `return -cfg.scale * state.welfare_true` is the most literal violation the rule describes, and
    it reaches a learner without touching an observation or an agent entry point -
    `enterprise_reward` is not one of the four names in `AGENT_INPUT_FUNCTIONS` - so a check scoped
    to `obs.py` plus those four functions passes it clean.
    """
    write(
        tree,
        "gosplan/env/reward.py",
        "def enterprise_reward(state, cfg):\n    return -cfg.scale * state.welfare_true\n",
    )
    found = guard.check_welfare_blindness(tree, None)
    assert rules(found) == ["6"]
    assert found[0].path == "gosplan/env/reward.py"
    assert "welfare_true" in found[0].message


def test_rule_6_welfare_in_the_ppo_package_fires(tree: Path) -> None:
    """The reward scope is where the reward is computed AND where it is consumed."""
    write(
        tree,
        "gosplan/agents/ppo/rollout.py",
        "def build_batch(rollout, cfg):\n    return rollout.rewards + rollout.val_measured\n",
    )
    found = guard.check_welfare_blindness(tree, None)
    assert rules(found) == ["6"]
    assert found[0].path == "gosplan/agents/ppo/rollout.py"


def test_rule_6_prose_in_the_reward_module_does_not_fire(tree: Path) -> None:
    """`gosplan/env/reward.py` quotes rule 6 in prose for the same reason it quotes rule 4."""
    write(
        tree,
        "gosplan/env/reward.py",
        '"""Reward terms (CONTRACT rules 4 and 6).\n'
        "\n"
        "welfare_true and val_measured are logged and never appear in any observation, reward or\n"
        'agent input; they are written through the ledger and read nowhere else."""\n'
        "\n"
        "# No welfare_true, no val_measured, no val_true in any term below.\n"
        'LOGGED_ONLY = ("welfare_true", "val_measured", "val_true", "welfare")\n'
        "\n"
        "def enterprise_reward(state, cfg):\n"
        '    """Exactly -scale * c_ik; no welfare term."""\n'
        "    return -reward_scale(cfg) * state.unit_cost\n",
    )
    assert guard.check_welfare_blindness(tree, None) == []


def test_rule_6_whole_file_scope_is_obs_plus_the_reward_scope(tree: Path) -> None:
    """The three destinations rule 6 names, as scopes: observation, reward, agent input."""
    for rel in (
        "gosplan/env/obs.py",
        "gosplan/env/reward.py",
        "gosplan/agents/ppo/adapter.py",
        "gosplan/agents/ppo/rollout.py",
        "gosplan/env/planner.py",
    ):
        write(tree, rel, "")
    scope = [guard._rel(tree, path) for path in guard._welfare_whole_file_scope(tree)]
    assert scope == [
        "gosplan/env/obs.py",
        "gosplan/env/reward.py",
        "gosplan/agents/ppo/adapter.py",
        "gosplan/agents/ppo/rollout.py",
    ]


def test_rule_6_whole_file_scope_is_not_double_reported(tree: Path) -> None:
    """A file scanned whole is skipped in the entry-point loop, so one hit is one violation."""
    write(
        tree,
        "gosplan/agents/ppo/adapter.py",
        "class Adapter:\n    def forward(self, obs, welfare_true):\n        return welfare_true\n",
    )
    found = guard.check_welfare_blindness(tree, None)
    assert rules(found) == ["6", "6"]  # the parameter and the return, once each
    assert {v.line for v in found} == {2, 3}


# -------------------------------------------------------------------------------------------------
# rule 7 - NO HARD-CODED PATHOLOGY (narrow)
# -------------------------------------------------------------------------------------------------


def test_rule_7_assignment_to_a_pathology_name_fires(tree: Path) -> None:
    """A transition rule that assigns to `padding` names the pathology it should let emerge."""
    write(
        tree,
        "gosplan/env/step.py",
        "def step(state, cfg):\n"
        "    padding = max(0.0, state.last_report - state.inv_output)\n"
        "    return padding\n",
    )
    found = guard.check_hardcoded_pathology(tree, None)
    assert rules(found) == ["7"]
    assert "assigned to" in found[0].message


def test_rule_7_branch_on_a_pathology_name_fires(tree: Path) -> None:
    """Branching on `hoard` is the same violation from the other direction."""
    write(
        tree,
        "gosplan/env/planner.py",
        "def allocate(view, cfg):\n"
        "    hoard = detect(view)\n"
        "    if hoard:\n"
        "        return penalise(view)\n"
        "    return view.requests\n",
    )
    found = guard.check_hardcoded_pathology(tree, None)
    assert set(rules(found)) == {"7"}
    assert any("branched on" in v.message for v in found)


def test_rule_7_prose_about_pathologies_does_not_fire(tree: Path) -> None:
    """Every module under `gosplan/env/` discusses padding and shaving; discussion is not a rule."""
    write(
        tree,
        "gosplan/env/planner.py",
        '"""Delivery: the padding-to-shortage channel (CONTRACT rule 7).\n'
        "\n"
        "A claim above stock lowers poolfill for the whole good; a claim below stock leaves the\n"
        "difference sitting in S. Padding, shaving, storming, hoarding and blat are CONSEQUENCES\n"
        'of the four lines below, never rules."""\n'
        "\n"
        "# padding and shaving must have nowhere to live except in the agent's policy\n"
        "def deliver(state, alloc, cfg):\n"
        '    """No branch here tests for padding or shaving."""\n'
        "    fill = min(1.0, state.inv_output / state.last_report)\n"
        "    return fill\n",
    )
    assert guard.check_hardcoded_pathology(tree, None) == []


def test_rule_7_note_states_that_passing_is_not_sufficient() -> None:
    """The guard prints rule 7's own wording on every run; this asserts it still says so."""
    assert "NECESSARY, NOT SUFFICIENT" in guard.RULE_7_NOTE
    assert "the lead reviews every env/ diff against this rule" in guard.RULE_7_NOTE


# -------------------------------------------------------------------------------------------------
# rule 9 - RNG
# -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        "import numpy as np\n\ndef f(cfg):\n    return np.random.normal(size=3)\n",
        "import numpy\n\ndef f(cfg):\n    return numpy.random.default_rng(0)\n",
        "import jax\n\ndef f(cfg):\n    return jax.random.normal(key)\n",
        "import random\n\ndef f(cfg):\n    return random.gauss(0.0, 1.0)\n",
        "from numpy.random import default_rng\n\ndef f(cfg):\n    return default_rng(0)\n",
    ],
)
def test_rule_9_direct_randomness_in_env_fires(tree: Path, source: str) -> None:
    """Every stream-based route into randomness under `gosplan/env/` is a rule-9 violation."""
    write(tree, "gosplan/env/production.py", source)
    found = guard.check_rng(tree, None)
    assert rules(found) and set(rules(found)) == {"9"}, source
    assert all("rng.draw" in v.fix for v in found)


def test_rule_9_keyed_draw_and_prose_do_not_fire(tree: Path) -> None:
    """`gosplan.rng.draw` is the sanctioned route, and the prose forbidding the others is exempt."""
    write(
        tree,
        "gosplan/env/production.py",
        '"""Production. CONTRACT rule 9: no numpy.random or jax.random call may appear anywhere\n'
        'under gosplan/env/; the yield draw goes through gosplan.rng.draw."""\n'
        "\n"
        "from gosplan.rng import draw\n"
        "\n"
        "# no numpy.random here, and no module-level default_rng generator\n"
        "def produce_step(state, cfg, t, k):\n"
        "    return draw(cfg.seed_env, 'yield', t, k)\n",
    )
    assert guard.check_rng(tree, None) == []


def test_rule_9_does_not_scan_the_agent_layer(tree: Path) -> None:
    """`seed_policy` is separate: the policy side may hold a Generator; this check ignores it."""
    write(
        tree,
        "gosplan/agents/heuristic.py",
        "import numpy as np\n\n"
        "class Random:\n"
        "    def act(self, obs, phase, rng: np.random.Generator):\n"
        "        return rng.uniform(size=obs.shape)\n",
    )
    assert guard.check_rng(tree, None) == []


# -------------------------------------------------------------------------------------------------
# rule 13 - TESTS ARE NOT EXPERIMENTS
# -------------------------------------------------------------------------------------------------


def test_rule_13_collectable_acceptance_file_fires(tree: Path) -> None:
    """A gate named so pytest would collect it is a rule-13 violation."""
    write(tree, "tests/acceptance/test_gate_g2.py", "def test_gate():\n    pass\n")
    write(tree, "tests/acceptance/gate_g2_phase1.py", "def run_gate():\n    pass\n")
    found = guard.check_acceptance_isolation(tree, None)
    assert rules(found) == ["13"]
    assert found[0].path == "tests/acceptance/test_gate_g2.py"


def test_rule_13_trailing_pattern_also_fires(tree: Path) -> None:
    """pytest's second default pattern, `*_test.py`, is checked too."""
    write(tree, "tests/acceptance/g3_test.py", "def test_gate():\n    pass\n")
    assert rules(guard.check_acceptance_isolation(tree, None)) == ["13"]


def test_rule_13_testpaths_naming_acceptance_fires(tree: Path) -> None:
    """`testpaths` must not name `tests/acceptance`; CI never collects a gate."""
    write(
        tree,
        "pyproject.toml",
        '[tool.pytest.ini_options]\ntestpaths = ["tests/unit", "tests/acceptance"]\n',
    )
    found = guard.check_acceptance_isolation(tree, None)
    assert rules(found) == ["13"]
    assert found[0].path == "pyproject.toml"


def test_rule_13_multiline_testpaths_array_fires(tree: Path) -> None:
    """THE SCRAPE-VERSUS-PARSE TEST: a multi-line array is valid TOML and must not slip through.

    A line-anchored regex captures only `[` here, so `tests/acceptance` was not found in it and the
    check passed *silently* - not even the "declares no testpaths" notice - on exactly the form a
    formatter produces from the single-line array.
    """
    write(
        tree,
        "pyproject.toml",
        "[tool.pytest.ini_options]\n"
        "testpaths = [\n"
        '    "tests/unit",\n'
        '    "tests/behavioural",\n'
        '    "tests/acceptance",\n'
        "]\n"
        'addopts = "-ra"\n',
    )
    found = guard.check_acceptance_isolation(tree, None)
    assert rules(found) == ["13"]
    assert found[0].path == "pyproject.toml"
    assert found[0].line == 2


def test_rule_13_testpaths_outside_the_pytest_table_is_not_testpaths(tree: Path) -> None:
    """`testpaths` means `[tool.pytest.ini_options] testpaths`, not any key of that name."""
    write(
        tree,
        "pyproject.toml",
        '[tool.something_else]\ntestpaths = ["tests/acceptance"]\n',
    )
    assert guard.check_acceptance_isolation(tree, None) == []


def test_rule_13_missing_testpaths_skips_with_a_notice(tree: Path, capsys) -> None:
    """No `testpaths` at all is a graceful skip, as the docstring promises."""
    write(tree, "pyproject.toml", '[project]\nname = "gosplan-env"\n')
    assert guard.check_acceptance_isolation(tree, None) == []
    assert "declares no testpaths" in capsys.readouterr().err


def test_rule_13_unparseable_pyproject_is_a_notice_not_a_crash(tree: Path, capsys) -> None:
    """A broken `pyproject.toml` is ruff's, uv's and pytest's finding, not a contract violation."""
    write(tree, "pyproject.toml", "[tool.pytest.ini_options\ntestpaths = [\n")
    assert guard.check_acceptance_isolation(tree, None) == []
    assert "does not parse as TOML" in capsys.readouterr().err


def test_rule_13_repository_testpaths_are_the_three_frozen_categories() -> None:
    """The live check: this repository collects PLAN section 11's three categories and no gate."""
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert guard._declared_testpaths(text) == ["tests/unit", "tests/behavioural", "tests/golden"]


def test_rule_13_gate_naming_convention_is_clean(tree: Path) -> None:
    """The convention already in the repository - `gate_<id>_<name>.py` - passes."""
    for name in ("gate_g0_mc_sanity.py", "gate_g1_regime_map.py", "gate_g4_final.py"):
        write(tree, f"tests/acceptance/{name}", "def run_gate():\n    pass\n")
    assert guard.check_acceptance_isolation(tree, None) == []


# -------------------------------------------------------------------------------------------------
# HELD-OUT PHENOMENA
# -------------------------------------------------------------------------------------------------


PHENOMENA_SKELETON = (
    '"""Phenomena. Rows 2, 5, 6 and 7 are HELD OUT until the Phase-2 acceptance run."""\n'
    "\n"
    "def phenomenon_bunching(ledger, cfg):\n"
    "{p1_bunching}"
    "\n"
    "def phenomenon_padding(ledger, cfg):\n"
    "{p1_padding}"
    "\n"
    "def phenomenon_storming(ledger, baseline_ledger, cfg):\n"
    "    return {{}}\n"
    "\n"
    "def phenomenon_hoarding(ledger, baseline_ledger, cfg):\n"
    "    return {{}}\n"
)


def test_held_out_call_from_a_phase1_experiment_fires(tree: Path) -> None:
    """The MC sanity harness may not compute a held-out row - WO-012's card forbids it in words."""
    write(
        tree,
        "gosplan/experiments/mc_sanity.py",
        "from gosplan.metrics.phenomena import phenomenon_hoarding\n"
        "\n"
        "def run(cfg):\n"
        "    return phenomenon_hoarding(ledger, baseline, cfg)\n",
    )
    found = guard.check_held_out_phenomena(tree, None)
    assert rules(found) == ["HELD-OUT"]
    assert "WO-030" in found[0].message
    assert "phenomenon_hoarding" in found[0].message


def test_held_out_call_from_the_phase1_section_of_phenomena_fires(tree: Path) -> None:
    """A Phase-1 row that reaches for a held-out row is the same violation."""
    write(
        tree,
        "gosplan/metrics/phenomena.py",
        PHENOMENA_SKELETON.format(
            p1_bunching="    return {}\n",
            p1_padding="    extra = phenomenon_hidden_reserves(ledger, cfg)\n    return extra\n",
        ),
    )
    found = guard.check_held_out_phenomena(tree, None)
    assert rules(found) == ["HELD-OUT"]
    assert "phenomenon_hidden_reserves" in found[0].message


def test_held_out_definitions_and_phase2_cross_references_do_not_fire(tree: Path) -> None:
    """Defining the held-out rows is WO-030's job; only calls from Phase-1 code are violations."""
    write(
        tree,
        "gosplan/metrics/phenomena.py",
        PHENOMENA_SKELETON.format(p1_bunching="    return {}\n", p1_padding="    return {}\n")
        + "\n"
        "def phenomenon_blat(ledger, cfg):\n"
        "    return phenomenon_hoarding(ledger, ledger, cfg)\n",
    )
    write(
        tree,
        "gosplan/experiments/phase1_gate.py",
        "from gosplan.metrics.phenomena import phenomenon_bunching\n"
        "\n"
        "def run(cfg):\n"
        "    return phenomenon_bunching(ledger, cfg)\n",
    )
    assert guard.check_held_out_phenomena(tree, None) == []


def test_held_out_set_is_exactly_rows_2_5_6_7() -> None:
    """Rows 1, 3 and 4 are pipeline checks and must never be added to the held-out set."""
    assert guard.HELD_OUT_FUNCTIONS == {
        "phenomenon_storming",
        "phenomenon_hoarding",
        "phenomenon_blat",
        "phenomenon_hidden_reserves",
    }
    assert guard.PHASE1_PHENOMENA == ("phenomenon_bunching", "phenomenon_padding")


# -------------------------------------------------------------------------------------------------
# PLAN LEAK
# -------------------------------------------------------------------------------------------------


# Assembled from fragments for the same reason the guard's own needles are: this file is tracked,
# and a literal here would make the repository fail its own absolute-path check.
LEAKY_PATHS = ("C:" + "\\Users\\someone\\dev", "/" + "home/someone/dev", "/" + "Users/someone/dev")


@pytest.mark.parametrize("leaked", LEAKY_PATHS)
def test_plan_leak_absolute_path_in_a_tracked_file_fires(tree: Path, leaked: str) -> None:
    """A local absolute path in a committed file is flagged wherever it appears."""
    write(tree, "docs/notes.md", f"The plan lives at {leaked}/gosplan-env/PLAN.md.\n")
    found = guard.check_plan_leak(tree, None)
    assert [v.rule for v in found] == ["PLAN-LEAK"], leaked
    assert found[0].path == "docs/notes.md"
    assert found[0].line == 1


def test_plan_leak_relative_paths_are_clean(tree: Path) -> None:
    """Repository-relative paths are what a machine-independent artefact uses."""
    write(tree, "docs/notes.md", "See gosplan/env/planner.py and tests/unit/test_planner.py.\n")
    assert guard.check_plan_leak(tree, None) == []


def test_plan_leak_untracked_plan_without_a_gitignore_entry_fires(tree: Path) -> None:
    """Without git, the fallback is `.gitignore`: PLAN.md must be named there."""
    write(tree, "PLAN.md", "# Build Plan\n")
    write(tree, ".gitignore", "__pycache__/\n")
    found = guard.check_plan_leak(tree, None)
    assert [v.rule for v in found] == ["PLAN-LEAK"]
    assert found[0].path == ".gitignore"


def test_plan_leak_gitignored_plan_is_clean(tree: Path) -> None:
    """A PLAN.md in the tree with a matching .gitignore entry is the intended state."""
    write(tree, "PLAN.md", "# Build Plan\n")
    write(tree, ".gitignore", "# supplied out of band; never commit it\nPLAN.md\n")
    assert guard.check_plan_leak(tree, None) == []


def test_plan_md_is_not_tracked_in_the_real_repository() -> None:
    """The live check, against the live repository: PLAN.md must not be in the index."""
    if not guard.git_available(REPO_ROOT):  # pragma: no cover - git is present in CI
        pytest.skip("git is unavailable")
    code, _ = guard._git(REPO_ROOT, "ls-files", "--error-unmatch", "PLAN.md")
    assert code != 0, "PLAN.md is tracked; it is supplied out of band and must never be committed"


# -------------------------------------------------------------------------------------------------
# formatting, CLI and the whole-tree run
# -------------------------------------------------------------------------------------------------


def test_github_format_is_a_workflow_command() -> None:
    """`::error file=PATH,line=N,title=Rule K::MSG`, with newlines escaped so nothing truncates."""
    v = guard.Violation(
        rule="4",
        path="gosplan/env/reward.py",
        line=42,
        message="first line\nsecond line",
        fix="delete it",
    )
    rendered = guard.format_github(v)
    assert rendered.startswith("::error file=gosplan/env/reward.py,line=42,title=Rule 4::")
    assert "\n" not in rendered
    assert "%0A" in rendered
    assert "fix: delete it" in rendered


def test_text_format_carries_the_rule_and_the_fix() -> None:
    """The human format names the file, the line, the rule and one line of remedy."""
    v = guard.Violation(
        rule="2", path="tests/unit/x.py", line=1, message="m", fix="f", severity="review-required"
    )
    rendered = guard.format_text(v)
    assert rendered.startswith("tests/unit/x.py:1: [rule 2] [review-required] m")
    assert "fix: f" in rendered


def test_format_auto_detects_github_actions(monkeypatch) -> None:
    """The format defaults to `github` under Actions and to `text` everywhere else."""
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert guard.resolve_format(None) == "github"
    assert guard.resolve_format("text") == "text"
    monkeypatch.setenv("GITHUB_ACTIONS", "")
    assert guard.resolve_format(None) == "text"
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    assert guard.resolve_format(None) == "text"


def test_run_all_reports_and_exits_non_zero_on_a_dirty_tree(tree: Path, capsys) -> None:
    """A tree with violations exits 1 and prints one entry per finding."""
    write(
        tree,
        "gosplan/env/planner.py",
        PLANNER_HEADER + "def update_targets(state: State, cfg):\n    ...\n",
    )
    assert guard.run_all(tree, None, "text") == 1
    out = capsys.readouterr().out
    assert "[rule 5]" in out
    assert guard.RULE_7_NOTE in out


def test_run_all_is_clean_on_a_clean_synthetic_tree(tree: Path) -> None:
    """The same tree with nothing wrong in it exits 0."""
    write(tree, "gosplan/env/planner.py", PLANNER_HEADER)
    assert guard.run_all(tree, None, "text") == 0


def test_real_repository_tree_passes_run_all(capsys) -> None:
    """THE REGRESSION ANCHOR: the repository as committed is clean under every whole-tree check.

    Diff-sensitive checks are skipped (`changed=None`) because they are properties of a change, not
    of a tree. If this fails, the change that broke it is the change to look at.
    """
    status = guard.run_all(REPO_ROOT, None, "text")
    captured = capsys.readouterr()
    assert status == 0, f"contract_guard reported violations on the repository:\n{captured.out}"


def test_every_check_accepts_the_common_signature(tree: Path) -> None:
    """Each check is `check_*(root, changed) -> list[Violation]`, callable on its own."""
    for label, check in guard.CHECKS:
        for changed in (None, {"gosplan/env/reward.py"}):
            result = check(tree, changed)
            assert isinstance(result, list), label
            assert all(isinstance(v, guard.Violation) for v in result), label


def test_cli_parses_the_documented_interface() -> None:
    """`--all`, `--base REF`, `--format {text,github}` and `--root PATH` are the interface."""
    parser = guard.build_parser()
    args = parser.parse_args(["--all", "--base", "origin/main", "--format", "github"])
    assert args.all is True
    assert args.base == "origin/main"
    assert args.format == "github"
    defaults = parser.parse_args([])
    assert defaults.all is False
    assert defaults.base is None
    assert defaults.format is None
    assert defaults.root is None


def test_main_on_the_real_repository_exits_zero(capsys) -> None:
    """End to end through `main`, the way CI invokes it."""
    assert guard.main(["--all", "--format", "text", "--root", str(REPO_ROOT)]) == 0
    capsys.readouterr()


def test_git_helpers_degrade_without_a_work_tree(tmp_path: Path) -> None:
    """No git work tree means the diff-sensitive checks are skipped, never crashed."""
    assert guard.git_available(tmp_path) is False
    assert guard.tracked_files(tmp_path) is None
    assert guard.changed_files(tmp_path, "origin/main") is None
