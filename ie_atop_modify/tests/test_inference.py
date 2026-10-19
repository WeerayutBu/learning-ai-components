import unittest
from application.inference import infer


class SpanTests(unittest.TestCase):
    def test_repeated_mentions_and_multi_token_names(self):
        relation = {"head": "New York", "tail": "France", "relation": "country", "relation_id": "P17"}
        seen = []
        def predict(document):
            seen.append(document)
            return [relation]
        result = infer("New York meets France.\n\nFrance greets New York.", "New York\nFrance", predict)
        row = result["relations"][0]
        self.assertEqual(row["head_spans"], [
            {"sentence": 0, "start": 0, "end": 2},
            {"sentence": 1, "start": 2, "end": 4}])
        self.assertEqual(row["tail_spans"], [
            {"sentence": 0, "start": 3, "end": 4},
            {"sentence": 1, "start": 0, "end": 1}])
        self.assertNotIn("head_spans", relation)
        for side in ("head", "tail"):
            for span in row[side + "_spans"]:
                tokens = seen[0]["sents"][span["sentence"]][span["start"]:span["end"]]
                self.assertEqual(" ".join(tokens), row[side])

    def test_unicode_and_no_relations(self):
        def predict(document):
            return [{"head": "東京", "tail": "日本"}]
        result = infer("東京、日本。", "東京\n日本", predict)
        self.assertEqual(result["relations"][0]["tail_spans"], [{"sentence": 0, "start": 2, "end": 3}])
        self.assertEqual(infer("東京、日本。", "東京\n日本", lambda doc: [])["relations"], [])


class ManualSpanTests(unittest.TestCase):
    def test_aliases_and_unicode_offsets(self):
        text = '😀 Acme met Beta.\n\nIt thanked Beta.'
        entities = [
            {'name': 'Acme', 'mentions': [{'start': 2, 'end': 6}, {'start': 18, 'end': 20}]},
            {'name': 'Beta', 'mentions': [{'start': 11, 'end': 15}]}]
        seen = []
        def predict(document):
            seen.append(document)
            return [{'head': 'Acme', 'tail': 'Beta'}]
        result = infer(text, entities, predict)
        self.assertEqual(seen[0]['vertexSet'][0][1]['pos'], [0, 1])
        self.assertEqual(result['relations'][0]['head_spans'], [
            {'sentence': 0, 'start': 1, 'end': 2},
            {'sentence': 1, 'start': 0, 'end': 1}])
        self.assertEqual(len(seen[0]['vertexSet'][1]), 1)

    def test_invalid_selections(self):
        for mention in [{'start': 1, 'end': 4}, {'start': 0, 'end': 10},
                        {'start': 0, 'end': 0}, {'start': False, 'end': 4},
                        {'start': 0, 'end': 99}]:
            with self.subTest(mention=mention), self.assertRaises(ValueError):
                infer('Acme\nBeta.', [
                    {'name': 'A', 'mentions': [mention]},
                    {'name': 'B', 'mentions': [{'start': 5, 'end': 9}]}],
                    lambda document: self.fail('Invalid spans reached model'))

    def test_duplicate_mention(self):
        with self.assertRaisesRegex(ValueError, 'only one entity'):
            infer('Acme Beta', [
                {'name': name, 'mentions': [{'start': 0, 'end': 4}]}
                for name in ['A', 'B']], lambda document: [])


class ExportSpanTests(unittest.TestCase):
    def test_all_mentions_export_without_relations(self):
        text = '😀 Acme met Beta.\n\nIt thanked Beta.'
        result = infer(text, [
            {'name': 'Acme', 'mentions': [{'start': 2, 'end': 6}, {'start': 18, 'end': 20}]},
            {'name': 'Beta', 'mentions': [{'start': 11, 'end': 15}]}], lambda document: [])
        self.assertEqual(result['text'], text)
        self.assertEqual(result['relations'], [])
        self.assertEqual(result['annotations'][0]['mentions'][1], {
            'mention': 'It', 'start': 18, 'end': 20,
            'token_span': {'sentence': 1, 'start': 0, 'end': 1}})
        self.assertEqual(len(result['annotations']), 2)

    def test_automatic_matching_exports_character_and_token_spans(self):
        text = 'New  York meets France.\nFrance visits New York.'
        result = infer(text, 'New York\nFrance', lambda document: [])
        self.assertEqual(result['annotations'][0]['mentions'][0], {
            'mention': 'New  York', 'start': 0, 'end': 9,
            'token_span': {'sentence': 0, 'start': 0, 'end': 2}})
        for entity in result['annotations']:
            self.assertEqual(len(entity['mentions']), 2)
            for mention in entity['mentions']:
                self.assertEqual(text[mention['start']:mention['end']], mention['mention'])
