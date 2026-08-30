"""Independent acceptance-harness tests; no checkpoint or network load."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/eval_procureagent.py"


def load_module():
    spec = importlib.util.spec_from_file_location("eval_procureagent", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_acceptance_core_passes_without_loading_checkpoint() -> None:
    module = load_module()
    report = module.run_acceptance(with_model=False, allow_missing_tesseract=True)
    assert report["passed"] is True
    assert report["live_model"] is None
    assert report["truth_boundary"] == {
        "simulation_only": True,
        "real_money_moved": False,
        "receipt_extractor": "tesseract_plus_deterministic_rules",
        "procurement_rl_policy_trained": False,
        "identity_router_dev_lab_trained": True,
        "identity_router_frozen_test_evaluated": False,
    }
    checks = {item["name"]: item for item in report["checks"]}
    assert checks["locked_contract"]["passed"]
    assert checks["three_axis_procuregym_comparison"]["passed"]
    assert checks["captured_identity_axis"]["passed"]
    if "ap_lifecycle_and_receipt_attacks" in checks:
        assert all(
            checks["ap_lifecycle_and_receipt_attacks"]["details"][
                "blocked_attacks"
            ].values()
        )


def test_cli_writes_valid_reproducible_json(tmp_path: Path) -> None:
    output = tmp_path / "acceptance.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--allow-missing-tesseract",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    stdout = json.loads(completed.stdout)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written == stdout
    assert stdout["schema_version"] == "procureagent-acceptance-v1"
    assert stdout["passed"] is True
    assert stdout["checks_passed"] == stdout["checks_total"]
