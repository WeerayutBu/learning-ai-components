import unittest
from application.inference import infer_document


class WebSpanTests(unittest.TestCase):
    def document(self):
        return {'document_id': 'repeat', 'text': '😀 Acme met Acme.', 'spans': [
            {'id': 'e0', 'kind': 'entity', 'label': 'ORG', 'start': 2, 'end': 6, 'text': 'Acme'},
            {'id': 'e1', 'kind': 'entity', 'label': 'ORG', 'start': 11, 'end': 15, 'text': 'Acme'}]}

    def test_repeated_names_stay_separate_and_offsets_survive_export(self):
        document = self.document()
        relation = {'head': 'e0', 'tail': 'e1', 'label': 'P749'}
        def predict(value):
            self.assertEqual(value['spans'], document['spans'])
            self.assertNotIn('vertexSet', value)
            self.assertNotIn('reviewed_pairs', value)
            return [relation]
        result = infer_document(document, predict)
        self.assertEqual(result, dict(document, relations=[relation]))

    def test_unknown_input_does_not_become_no_relation(self):
        result = infer_document(dict(self.document(), reviewed_pairs=[{'head': 'e0', 'tail': 'e1'}]), lambda doc: [])
        self.assertEqual(result['relations'], [])
        self.assertNotIn('reviewed_pairs', result)

    def test_invalid_offsets_text_and_ids_rejected(self):
        for change in ({'start': True}, {'start': 3}, {'text': 'wrong'}, {'id': 'e0'}, {'kind': 'other'}):
            document = self.document()
            document['spans'][1].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                infer_document(document, lambda doc: self.fail('Invalid input reached predictor'))
