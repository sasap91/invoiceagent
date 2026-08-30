"""Dependency-light smoke tests for the synthetic Streamlit demo."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from demo import LEDGER, PAYMENT_STORY, SCENARIOS, SCENARIO_ORDER, route_structured_entry


ROOT = Path(__file__).resolve().parents[1]


class DemoFixtureTests(unittest.TestCase):
    def test_payment_story_reconciles_500_to_200_to_zero(self) -> None:
        self.assertEqual(
            [stage["outstanding"] for stage in PAYMENT_STORY],
            [500.0, 200.0, 0.0],
        )
        self.assertEqual(PAYMENT_STORY[-1]["status"], "Paid")
        self.assertEqual(
            sum(amount for _, amount in PAYMENT_STORY[-1]["receipts"]),
            500.0,
        )

    def test_three_required_routes_are_present(self) -> None:
        self.assertEqual(len(SCENARIO_ORDER), 3)
        self.assertEqual(
            [SCENARIOS[key]["route"] for key in SCENARIO_ORDER],
            ["local", "small_model", "escalate"],
        )

    def test_every_scenario_discloses_fixture_provenance(self) -> None:
        for key in SCENARIO_ORDER:
            notice = SCENARIOS[key]["fixture_notice"].casefold()
            self.assertIn("fixture", notice)
            self.assertIn("no ocr", notice)
            self.assertIn("no", notice)
            self.assertIn("model", notice)

    def test_model_scenarios_are_replays(self) -> None:
        for key in ("model-accept", "ambiguous-escalate"):
            replay = SCENARIOS[key]["model_replay"]
            self.assertIsNotNone(replay)
            self.assertIn("not generated", replay["provenance"].casefold())

    def test_ledger_contains_both_books_and_review_state(self) -> None:
        self.assertEqual({row["kind"] for row in LEDGER}, {"AP", "AR"})
        self.assertTrue(any(row["reconciliation"] == "Needs review" for row in LEDGER))


class StructuredFallbackTests(unittest.TestCase):
    def test_zero_dollar_pair_is_rejected(self) -> None:
        result = route_structured_entry(
            {
                "invoice_id": "A-10",
                "invoice_amount": 0,
                "payment_reference": "A-10",
                "payment_amount": 0,
                "known_party": True,
            }
        )
        self.assertEqual(result["route"], "escalate")
        self.assertFalse(result["accepted"])

    def test_exact_known_match_is_local(self) -> None:
        result = route_structured_entry(
            {
                "invoice_id": "A-10",
                "invoice_amount": "125.00",
                "payment_reference": "a-10",
                "payment_amount": "125.00",
                "known_party": True,
            }
        )
        self.assertEqual(result["route"], "local")
        self.assertTrue(result["accepted"])
        self.assertFalse(result["model_called"])

    def test_missing_reference_does_not_fabricate_model_capability(self) -> None:
        result = route_structured_entry(
            {
                "invoice_id": "A-10",
                "invoice_amount": 125,
                "payment_reference": "memo only",
                "payment_amount": 125,
                "known_party": True,
            }
        )
        self.assertEqual(result["route"], "escalate")
        self.assertFalse(result["accepted"])
        self.assertFalse(result["model_called"])
        self.assertIn("does not invent", result["reason"])

    def test_conflicting_amount_escalates(self) -> None:
        result = route_structured_entry(
            {
                "invoice_id": "A-10",
                "invoice_amount": 125,
                "payment_reference": "A-10",
                "payment_amount": 100,
                "known_party": True,
            }
        )
        self.assertEqual(result["route"], "escalate")
        self.assertFalse(result["accepted"])


class AppSourceSmokeTests(unittest.TestCase):
    def test_app_parses_without_importing_streamlit(self) -> None:
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        ast.parse(source)

    def test_app_contains_integrity_and_manual_entry_contracts(self) -> None:
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("synthetic fixture replay", source.casefold())
        self.assertIn("no document was scanned", source.casefold())
        self.assertIn("route_structured_entry", source)
        self.assertIn("Invoice & payment ledger", source)


if __name__ == "__main__":
    unittest.main()
