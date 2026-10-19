import unittest
import torch
from infrastructure.prepro import read_docred, docred_rel2id
from infrastructure.losses import ATLoss


class Tokenizer:
    def tokenize(self, token):
        return [token]
    def convert_tokens_to_ids(self, tokens):
        return list(range(len(tokens)))
    def build_inputs_with_special_tokens(self, ids):
        return [101] + ids + [102]


class SpanPreprocessingTests(unittest.TestCase):
    def document(self):
        return {'document_id': 'test', 'text': '😀 Acme meets Acme.', 'spans': [
            {'id': 'a', 'kind': 'entity', 'label': 'ORG', 'start': 2, 'end': 6, 'text': 'Acme'},
            {'id': 'b', 'kind': 'entity', 'label': 'ORG', 'start': 13, 'end': 17, 'text': 'Acme'}],
            'relations': [{'head': 'a', 'tail': 'b', 'label': 'P127'}]}

    def test_repeated_names_stay_separate(self):
        feature, = read_docred(self.document(), Tokenizer())
        self.assertEqual(feature['hts'], [[0, 1], [1, 0]])
        self.assertEqual(feature['entity_pos'], [[(1, 4)], [(5, 8)]])
        self.assertEqual(feature['labels'][0][docred_rel2id['P127']], 1)
        self.assertTrue(all(v == -1 for v in feature['labels'][1]))
        self.assertEqual(feature['labels'][0][docred_rel2id['P17']], -1)

    def test_invalid_span_or_relation_is_rejected(self):
        for change in ['offset', 'text', 'id', 'reference']:
            document = self.document()
            if change == 'offset':document['spans'][0]['start'] = 3
            if change == 'text':document['spans'][0]['text'] = 'Other'
            if change == 'id':document['spans'][1]['id'] = 'a'
            if change == 'reference':document['relations'][0]['tail'] = 'missing'
            with self.subTest(change=change), self.assertRaises(ValueError):
                read_docred(document, Tokenizer())

    def test_unknown_labels_have_zero_gradient(self):
        logits = torch.tensor([[0., 1., 2.], [1., 2., 3.]], requires_grad=True)
        labels = torch.tensor([[-1., 1., -1.], [-1., -1., -1.]])
        original = labels.clone()
        loss = ATLoss()(logits, labels)
        loss.backward()
        self.assertGreater(loss.item(), 0)
        self.assertEqual(logits.grad[0, 2].item(), 0)
        self.assertEqual(logits.grad[1].abs().sum().item(), 0)
        torch.testing.assert_close(labels, original)

    def test_all_unknown_loss_is_differentiable_zero(self):
        logits = torch.ones(2, 3, requires_grad=True)
        loss = ATLoss()(logits, torch.full((2, 3), -1.))
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertEqual(logits.grad.abs().sum().item(), 0)

    def test_reviewed_missing_pair_generates_no_relation(self):
        document = self.document()
        document['reviewed_pairs'] = [{'head': 'b', 'tail': 'a'}]
        feature, = read_docred(document, Tokenizer())
        self.assertEqual(feature['labels'][1], [1] + [0] * (len(docred_rel2id) - 1))
        self.assertEqual(feature['labels'][0][docred_rel2id['P17']], -1)

    def test_complete_annotations_generate_negatives_but_preserve_positives(self):
        document = self.document()
        document['relations_complete'] = True
        feature, = read_docred(document, Tokenizer())
        self.assertEqual(feature['labels'][0][docred_rel2id['P127']], 1)
        self.assertEqual(feature['labels'][0][docred_rel2id['P17']], 0)
        self.assertEqual(feature['labels'][1][0], 1)

    def test_unlabeled_test_is_unknown(self):
        document = self.document()
        document['relations'] = []
        feature, = read_docred(document, Tokenizer())
        self.assertTrue(all(label == -1 for row in feature['labels'] for label in row))

    def test_reviewed_pair_validation(self):
        document = self.document()
        document['reviewed_pairs'] = [{'head': 'missing', 'tail': 'a'}]
        with self.assertRaisesRegex(ValueError, 'reviewed pair'):
            read_docred(document, Tokenizer())

    def test_original_whitespace_offsets_are_preserved(self):
        document = self.document()
        document['spans'][0].update(start=1, end=7, text=' Acme ')
        feature, = read_docred(document, Tokenizer())
        self.assertEqual(feature['entity_pos'][0], [(1, 4)])
        self.assertEqual(feature['document']['spans'][0]['text'], ' Acme ')
