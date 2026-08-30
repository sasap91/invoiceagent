from decimal import Decimal
import importlib
import os
import unittest
from unittest.mock import patch

from invoiceagent import (
    OcrDocument,
    RoutingAction,
    decide_small_first_route,
)
from invoiceagent.core import ValidationError
from invoiceagent.extraction import (
    NOT_EVALUATED_LABEL,
    EntitySpan,
    InvoiceNumberResult,
    TokenPrediction,
    assemble_token_predictions,
    decode_bio_spans,
    extract_anchored_identifier,
    is_valid_invoice_identifier,
    normalize_pixel_boxes,
    validate_token_prediction_alignment,
)


class OcrTests(unittest.TestCase):
    def test_pixel_boxes_are_normalized(self):
        self.assertEqual(
            normalize_pixel_boxes([(10, 20, 30, 60)], 100, 100),
            ((100, 200, 300, 600),),
        )

    def test_ocr_requires_aligned_words_and_boxes(self):
        with self.assertRaises(ValueError):
            OcrDocument(["Invoice"], [], "0.9")


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.words = ("Invoice", "No", "INV", "-", "204", "Total", "500.00")
        self.boxes = tuple((index * 100, 10, index * 100 + 80, 60) for index in range(7))

    def test_anchored_rule_and_identifier_validation(self):
        self.assertEqual(extract_anchored_identifier(self.words), "INV-204")
        self.assertTrue(is_valid_invoice_identifier("INV-204"))
        self.assertFalse(is_valid_invoice_identifier("2026/08/30"))
        self.assertFalse(is_valid_invoice_identifier("500.00"))

    def test_bio_decoder_scores_only_entity_tokens(self):
        labels = ("O", "O", "LABEL_1", "LABEL_2", "LABEL_2", "O", "O")
        confidences = ("0.999", "0.999", "0.96", "0.93", "0.95", "0.999", "0.999")
        margins = ("0.99", "0.99", "0.70", "0.60", "0.65", "0.99", "0.99")
        spans = decode_bio_spans(self.words, self.boxes, labels, confidences, margins)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].value, "INV-204")
        self.assertEqual(spans[0].minimum_confidence, Decimal("0.93"))
        self.assertNotEqual(spans[0].mean_confidence, Decimal("0.999"))

    def test_multiple_disjoint_spans_are_ambiguous(self):
        labels = ("O", "O", "B-INVOICE", "O", "O", "B-INVOICE", "O")
        spans = decode_bio_spans(
            self.words,
            self.boxes,
            labels,
            ("0.9",) * 7,
            ("0.5",) * 7,
        )
        self.assertEqual(len(spans), 2)

    def test_orphan_inside_tag_is_rejected(self):
        spans = decode_bio_spans(
            self.words,
            self.boxes,
            ("O", "O", "I-INVOICE", "O", "O", "O", "O"),
            ("0.9",) * 7,
            ("0.5",) * 7,
        )
        self.assertEqual(spans, ())

    def test_unrelated_claim_is_not_grounded_by_valid_indices(self):
        fake = EntitySpan(
            value="FAKE-999",
            word_indices=(2, 3, 4),
            boxes=self.boxes[2:5],
            token_confidences=(Decimal("0.99"),) * 3,
            token_margins=(Decimal("0.8"),) * 3,
        )
        result = InvoiceNumberResult((fake,), "ryan-adapter", Decimal("42.0"))
        ocr = OcrDocument(self.words, self.boxes, "0.92")
        signals = result.routing_signals(
            ocr, heuristic_candidate=None, escalation_available=False
        )
        self.assertFalse(signals.extraction.grounded)
        self.assertEqual(
            decide_small_first_route(signals).action,
            RoutingAction.HUMAN_REVIEW,
        )

    def test_result_routes_grounded_agreement(self):
        span = EntitySpan(
            value="INV-204",
            word_indices=(2, 3, 4),
            boxes=self.boxes[2:5],
            token_confidences=(Decimal("0.96"), Decimal("0.95"), Decimal("0.97")),
            token_margins=(Decimal("0.7"), Decimal("0.6"), Decimal("0.7")),
        )
        result = InvoiceNumberResult((span,), "ryan-adapter", Decimal("42.0"))
        ocr = OcrDocument(self.words, self.boxes, "0.92")
        signals = result.routing_signals(
            ocr, heuristic_candidate="inv204", escalation_available=True
        )
        decision = decide_small_first_route(signals)
        self.assertEqual(decision.action, RoutingAction.ACCEPT)

    def test_missing_span_goes_to_human_when_escalation_unavailable(self):
        result = InvoiceNumberResult((), "ryan-adapter", Decimal("41.0"))
        ocr = OcrDocument(self.words, self.boxes, "0.92")
        decision = decide_small_first_route(
            result.routing_signals(
                ocr, heuristic_candidate=None, escalation_available=False
            )
        )
        self.assertEqual(decision.action, RoutingAction.HUMAN_REVIEW)


class TokenPredictionValidationTests(unittest.TestCase):
    def test_negative_index_is_rejected(self):
        with self.assertRaises(ValidationError):
            TokenPrediction(
                index=-1, word="Invoice", box=(0, 0, 10, 10),
                label="O", confidence=Decimal("0.9"), margin=Decimal("0.1"),
            )

    def test_non_integer_index_is_rejected(self):
        with self.assertRaises(ValidationError):
            TokenPrediction(
                index=1.5, word="Invoice", box=(0, 0, 10, 10),
                label="O", confidence=Decimal("0.9"), margin=Decimal("0.1"),
            )

    def test_empty_word_is_rejected(self):
        with self.assertRaises(ValidationError):
            TokenPrediction(
                index=0, word="", box=(0, 0, 10, 10),
                label="O", confidence=Decimal("0.9"), margin=Decimal("0.1"),
            )

    def test_malformed_box_is_rejected(self):
        with self.assertRaises(ValidationError):
            TokenPrediction(
                index=0, word="Invoice", box=(0, 0, 10),
                label="O", confidence=Decimal("0.9"), margin=Decimal("0.1"),
            )

    def test_out_of_range_confidence_is_rejected(self):
        with self.assertRaises(ValidationError):
            TokenPrediction(
                index=0, word="Invoice", box=(0, 0, 10, 10),
                label="O", confidence=Decimal("1.5"), margin=Decimal("0.1"),
            )

    def test_out_of_range_margin_is_rejected(self):
        with self.assertRaises(ValidationError):
            TokenPrediction(
                index=0, word="Invoice", box=(0, 0, 10, 10),
                label="O", confidence=Decimal("0.9"), margin=Decimal("-0.1"),
            )

    def test_well_formed_prediction_reports_evaluated(self):
        prediction = TokenPrediction(
            index=3, word="INV-204", box=(0, 0, 10, 10),
            label="LABEL_1", confidence=Decimal("0.9"), margin=Decimal("0.4"),
        )
        self.assertTrue(prediction.evaluated)

    def test_not_evaluated_label_reports_unevaluated(self):
        prediction = TokenPrediction(
            index=3, word="INV-204", box=(0, 0, 10, 10),
            label=NOT_EVALUATED_LABEL, confidence=Decimal("0"), margin=Decimal("0"),
        )
        self.assertFalse(prediction.evaluated)


class TruncationTests(unittest.TestCase):
    """Regression coverage for tokenizer-truncated input.

    Reproduces the reported bug: a 600-word document where the tokenizer's
    max_length truncation means the model never scores word indices from 255
    onward. Those words must be reported as NOT_EVALUATED_LABEL, not as
    invented "O" background predictions with fabricated confidence/margin.
    """

    def test_words_past_the_truncation_cutoff_are_marked_not_evaluated(self):
        word_count = 600
        truncated_at = 255
        words = tuple(f"word{i}" for i in range(word_count))
        boxes = tuple((i, 0, i + 1, 10) for i in range(word_count))
        per_word = {
            index: ("LABEL_0", Decimal("0.98"), Decimal("0.9"))
            for index in range(truncated_at)
        }

        predictions = assemble_token_predictions(words, boxes, per_word)

        self.assertEqual(len(predictions), word_count)
        evaluated = predictions[:truncated_at]
        truncated = predictions[truncated_at:]
        self.assertEqual(len(truncated), word_count - truncated_at)
        self.assertTrue(all(prediction.evaluated for prediction in evaluated))
        self.assertTrue(all(not prediction.evaluated for prediction in truncated))
        self.assertTrue(all(prediction.label == NOT_EVALUATED_LABEL for prediction in truncated))
        # Explicit OCR indices are retained even for truncated words.
        for expected_index, prediction in enumerate(predictions):
            self.assertEqual(prediction.index, expected_index)
            self.assertEqual(prediction.word, words[expected_index])
            self.assertEqual(prediction.box, boxes[expected_index])
        # Truncated words never enter a decoded span: they behave as background,
        # exactly like a real "O" prediction, for entity-detection purposes.
        labels = [prediction.label for prediction in predictions]
        confidences = [prediction.confidence for prediction in predictions]
        margins = [prediction.margin for prediction in predictions]
        spans = decode_bio_spans(words, boxes, labels, confidences, margins)
        self.assertEqual(spans, ())

    def test_fully_evaluated_input_has_no_not_evaluated_predictions(self):
        words = ("Invoice", "No", "INV-204")
        boxes = ((0, 0, 10, 10), (10, 0, 20, 10), (20, 0, 30, 10))
        per_word = {
            0: ("O", Decimal("0.99"), Decimal("0.9")),
            1: ("O", Decimal("0.99"), Decimal("0.9")),
            2: ("LABEL_1", Decimal("0.9"), Decimal("0.4")),
        }
        predictions = assemble_token_predictions(words, boxes, per_word)
        self.assertTrue(all(prediction.evaluated for prediction in predictions))


class AlignmentValidationTests(unittest.TestCase):
    def setUp(self):
        self.words = ("Invoice", "No", "INV-204")
        self.boxes = ((0, 0, 10, 10), (10, 0, 20, 10), (20, 0, 30, 10))
        self.predictions = assemble_token_predictions(
            self.words,
            self.boxes,
            {0: ("O", Decimal("0.9"), Decimal("0.1")), 1: ("O", Decimal("0.9"), Decimal("0.1"))},
        )

    def test_well_aligned_predictions_pass(self):
        validate_token_prediction_alignment(self.predictions, self.words, self.boxes)

    def test_wrong_length_is_rejected(self):
        with self.assertRaises(ValidationError):
            validate_token_prediction_alignment(self.predictions[:-1], self.words, self.boxes)

    def test_reordered_predictions_are_rejected(self):
        reordered = (self.predictions[1], self.predictions[0], self.predictions[2])
        with self.assertRaises(ValidationError):
            validate_token_prediction_alignment(reordered, self.words, self.boxes)

    def test_mismatched_word_is_rejected(self):
        tampered = self.predictions[0]
        swapped = TokenPrediction(
            index=tampered.index, word="SomethingElse", box=tampered.box,
            label=tampered.label, confidence=tampered.confidence, margin=tampered.margin,
        )
        mismatched = (swapped,) + self.predictions[1:]
        with self.assertRaises(ValidationError):
            validate_token_prediction_alignment(mismatched, self.words, self.boxes)

    def test_mismatched_box_is_rejected(self):
        tampered = self.predictions[0]
        swapped = TokenPrediction(
            index=tampered.index, word=tampered.word, box=(99, 99, 199, 199),
            label=tampered.label, confidence=tampered.confidence, margin=tampered.margin,
        )
        mismatched = (swapped,) + self.predictions[1:]
        with self.assertRaises(ValidationError):
            validate_token_prediction_alignment(mismatched, self.words, self.boxes)


class EnvironmentOverrideTests(unittest.TestCase):
    """DEFAULT_ADAPTER_MODEL and friends must honor env var overrides at import time."""

    def _reload_with_env(self, env: dict[str, str]):
        with patch.dict(os.environ, env, clear=False):
            import invoiceagent.extraction as extraction_module

            return importlib.reload(extraction_module)

    def tearDown(self):
        # Always reload once more with a clean environment so later tests in
        # this process see the real defaults again.
        import invoiceagent.extraction as extraction_module

        importlib.reload(extraction_module)

    def test_adapter_model_override_is_honored(self):
        module = self._reload_with_env(
            {"INVOICEAGENT_ADAPTER_MODEL": "/local/checkout/adapter"}
        )
        self.assertEqual(module.DEFAULT_ADAPTER_MODEL, "/local/checkout/adapter")

    def test_base_model_override_is_honored(self):
        module = self._reload_with_env({"INVOICEAGENT_BASE_MODEL": "local/base-model"})
        self.assertEqual(module.DEFAULT_BASE_MODEL, "local/base-model")

    def test_empty_adapter_revision_override_becomes_none(self):
        module = self._reload_with_env({"INVOICEAGENT_ADAPTER_REVISION": ""})
        self.assertIsNone(module.DEFAULT_ADAPTER_REVISION)

    def test_empty_base_revision_override_becomes_none(self):
        module = self._reload_with_env({"INVOICEAGENT_BASE_REVISION": ""})
        self.assertIsNone(module.DEFAULT_BASE_REVISION)

    def test_unset_env_vars_keep_the_public_hugging_face_defaults(self):
        with patch.dict(os.environ, {}, clear=False):
            for key in (
                "INVOICEAGENT_ADAPTER_MODEL",
                "INVOICEAGENT_BASE_MODEL",
                "INVOICEAGENT_ADAPTER_REVISION",
                "INVOICEAGENT_BASE_REVISION",
            ):
                os.environ.pop(key, None)
            import invoiceagent.extraction as extraction_module

            module = importlib.reload(extraction_module)
        self.assertEqual(module.DEFAULT_ADAPTER_MODEL, "ryanznie/layoutlmv3-lora-invoice-number")
        self.assertEqual(module.DEFAULT_BASE_MODEL, "microsoft/layoutlmv3-base")
        self.assertEqual(
            module.DEFAULT_ADAPTER_REVISION, "7dc28f5a3b14aa100ba432ee1b0a6cac6c7b2c5c"
        )
        self.assertEqual(module.DEFAULT_BASE_REVISION, "cfbbbff0762e6aab37086fdd4739ad14fe7d5db4")


if __name__ == "__main__":
    unittest.main()
