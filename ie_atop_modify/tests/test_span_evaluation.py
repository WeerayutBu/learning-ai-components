import unittest
import numpy as np
from infrastructure.evaluation import evaluate_spans, span_predictions
from infrastructure.prepro import read_docred, docred_rel2id
import test_span_preprocessing as fixtures


class SpanEvaluationTests(unittest.TestCase):
    def test_masked_precision_and_recall(self):
        document = fixtures.SpanPreprocessingTests().document()
        document['reviewed_pairs'] = [{'head': 'b', 'tail': 'a'}]
        features = read_docred(document, fixtures.Tokenizer())
        preds = np.zeros((2, len(docred_rel2id)))
        preds[0, docred_rel2id['P127']] = 1  # Known true positive.
        preds[0, docred_rel2id['P17']] = 1   # Unknown: exclude from score.
        preds[1, docred_rel2id['P17']] = 1   # Reviewed negative: false positive.
        documents = span_predictions(preds, features)
        self.assertEqual(documents[0]['spans'], document['spans'])
        self.assertEqual(documents[0]['relations'][0]['head'], 'a')
        metrics = evaluate_spans(documents, features)
        self.assertEqual(metrics['precision'], 0.5)
        self.assertEqual(metrics['recall'], 1)
        self.assertAlmostEqual(metrics['f1'], 2 / 3)
        self.assertEqual(metrics['unknown_predictions'], 1)

    def test_empty_predictions_and_unlabeled_dev(self):
        document = fixtures.SpanPreprocessingTests().document()
        features = read_docred(document, fixtures.Tokenizer())
        documents = span_predictions(np.zeros((2, len(docred_rel2id))), features)
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0]['relations'], [])
        self.assertEqual(evaluate_spans(documents, features)['f1'], 0)
        document['relations'] = []
        features = read_docred(document, fixtures.Tokenizer())
        with self.assertRaisesRegex(ValueError, 'known positive'):
            evaluate_spans(documents, features)
