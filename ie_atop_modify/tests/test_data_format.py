import json
from pathlib import Path
import tempfile
import unittest

from infrastructure.data_format import convert_document, read_documents


class DataFormatTests(unittest.TestCase):
    def setUp(self):
        self.document = {'title': 'Synthetic', 'sents': [['😀', 'Acme'], ['Acme', 'Paris']],
                         'vertexSet': [[{'sent_id': 0, 'pos': [1, 2], 'type': 'ORG'},
                                        {'sent_id': 1, 'pos': [0, 1], 'type': 'ORG'}],
                                       [{'sent_id': 1, 'pos': [1, 2], 'type': 'LOC'}]],
                         'labels': [{'h': 0, 't': 1, 'r': 'P159'}]}

    def test_separate_mentions_and_unicode_offsets(self):
        result, omitted = convert_document(self.document, 'expand')
        self.assertEqual(len(result['spans']), 3)
        self.assertEqual(result['spans'][0]['start'], 2)
        for span in result['spans']:
            self.assertEqual(result['text'][span['start']:span['end']], span['text'])
        self.assertEqual(result['relations'], [
            {'head': 'e0', 'tail': 'e2', 'label': 'P159'},
            {'head': 'e1', 'tail': 'e2', 'label': 'P159'}])
        self.assertEqual(omitted, 0)

    def test_ambiguous_group_labels_are_counted(self):
        result, omitted = convert_document(self.document, 'singletons')
        self.assertEqual(result['relations'], [])
        self.assertEqual(omitted, 1)
        self.assertEqual(len(result['spans']), 3)

    def test_stream_across_buffer_boundaries(self):
        document = dict(self.document, title='x' * 70000)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.json'
            path.write_text(json.dumps([document, self.document]))
            self.assertEqual(list(read_documents(path)), [document, self.document])

    def test_coverage_only_for_singletons_in_labeled_splits(self):
        self.document['vertexSet'].append([{'sent_id': 0, 'pos': [0, 1], 'type': 'MISC'}])
        result, _ = convert_document(self.document, 'singletons')
        self.assertEqual(result['reviewed_pairs'], [{'head': 'e2', 'tail': 'e3'}, {'head': 'e3', 'tail': 'e2'}])
        self.assertEqual(result['relations'], [])
        del self.document['labels']
        result, _ = convert_document(self.document, 'singletons')
        self.assertEqual(result['reviewed_pairs'], [])
