from decimal import Decimal
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import invoiceagent.extraction as extraction

from invoiceagent import (
    OcrDocument,
    RoutingAction,
    decide_small_first_route,
)
from invoiceagent.extraction import (
    EntitySpan,
    InvoiceNumberResult,
    LayoutLMv3InvoiceExtractor,
    TokenPrediction,
    decode_bio_spans,
    extract_anchored_identifier,
    is_valid_invoice_identifier,
    normalize_pixel_boxes,
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


class ConfigurationTests(unittest.TestCase):
    def test_local_model_overrides_and_blank_revision_are_explicit(self):
        with patch.dict(
            "os.environ",
            {
                "INVOICEAGENT_ADAPTER_MODEL": "/private/models/layoutlmv3-local",
                "INVOICEAGENT_ADAPTER_REVISION": "",
            },
        ):
            self.assertEqual(
                extraction._configured_model("INVOICEAGENT_ADAPTER_MODEL", "fallback"),
                "/private/models/layoutlmv3-local",
            )
            self.assertIsNone(
                extraction._configured_revision("INVOICEAGENT_ADAPTER_REVISION", "pinned")
            )
        self.assertEqual(
            extraction.public_model_ref("/private/models/layoutlmv3-local", None),
            "layoutlmv3-local",
        )


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

    def test_token_prediction_contract_rejects_untrusted_values(self):
        with self.assertRaisesRegex(ValueError, "word_index"):
            TokenPrediction(-1, "Invoice", (0, 0, 10, 10), "O", Decimal("0.9"), Decimal("0.2"))
        with self.assertRaisesRegex(ValueError, "box"):
            TokenPrediction(0, "Invoice", (10, 10, 0, 0), "O", Decimal("0.9"), Decimal("0.2"))
        with self.assertRaisesRegex(ValueError, "confidence"):
            TokenPrediction(0, "Invoice", (0, 0, 10, 10), "O", Decimal("9"), Decimal("0.2"))

    def test_result_rejects_duplicate_or_reordered_prediction_indices(self):
        first = TokenPrediction(
            0, "Invoice", (0, 0, 80, 60), "O", Decimal("0.9"), Decimal("0.4")
        )
        second = TokenPrediction(
            1, "No", (100, 10, 180, 60), "O", Decimal("0.8"), Decimal("0.3")
        )
        with self.assertRaisesRegex(ValueError, "unique increasing"):
            InvoiceNumberResult((), "model", Decimal("1"), (second, first))

    def test_predict_emits_only_tokens_the_tokenizer_actually_evaluated(self):
        class FakeTensor:
            def to(self, _device):
                return self

        class FakeEncoding(dict):
            def __init__(self):
                super().__init__(input_ids=FakeTensor())

            def word_ids(self, _batch_index):
                return (None, 0, 1, None)

        class FakeProcessor:
            def __call__(self, *_args, **_kwargs):
                return FakeEncoding()

        class FakeScalar:
            def __init__(self, value):
                self.value = value

            def detach(self):
                return self

            def cpu(self):
                return self

            def item(self):
                return self.value

        class FakeTopK:
            def __init__(self, rows):
                self.rows = rows

            def __getitem__(self, key):
                row, column = key
                return FakeScalar(self.rows[row][column])

        class FakeProbabilities:
            def topk(self, *, k, dim):
                self.assertions = (k, dim)
                return (
                    FakeTopK(((0.9, 0.1), (0.92, 0.08), (0.95, 0.05), (0.9, 0.1))),
                    FakeTopK(((0, 1), (0, 1), (1, 0), (0, 1))),
                )

        class FakeTorch:
            def no_grad(self):
                return nullcontext()

            def softmax(self, _logits, *, dim):
                self.softmax_dim = dim
                return FakeProbabilities()

        class FakeModel:
            config = SimpleNamespace(id2label={0: "O", 1: "B-INVOICE_ID"})

            def __call__(self, **_kwargs):
                return SimpleNamespace(logits=[object()])

        ocr = OcrDocument(
            ("Invoice", "FF-10482", "TRUNCATED-A", "TRUNCATED-B"),
            ((0, 0, 100, 50), (110, 0, 250, 50), (0, 60, 150, 110), (160, 60, 320, 110)),
            Decimal("0.95"),
        )
        extractor = LayoutLMv3InvoiceExtractor(
            adapter_model="/Users/example/private/layoutlmv3-adapter",
            adapter_revision=None,
            device="cpu",
        )
        extractor.processor = FakeProcessor()
        extractor.model = FakeModel()
        extractor._torch = FakeTorch()
        extractor.device = "cpu"

        result = extractor.predict(object(), ocr)

        self.assertEqual(
            tuple(item.word_index for item in result.token_predictions), (0, 1)
        )
        self.assertEqual(
            tuple(item.word for item in result.token_predictions),
            ("Invoice", "FF-10482"),
        )
        self.assertEqual(result.spans[0].word_indices, (1,))
        self.assertNotIn("TRUNCATED", " ".join(item.word for item in result.token_predictions))
        self.assertNotIn("/Users/example/private", result.model_name)
        self.assertIn("layoutlmv3-adapter", result.model_name)


if __name__ == "__main__":
    unittest.main()
