from pathlib import Path
import re

from infrastructure.data_format import read_documents
from tqdm import tqdm
import ujson as json

docred_rel2id = json.loads((Path(__file__).resolve().parents[2] / "data/rel2id.json").read_text())


def read_docred(file_in, tokenizer, max_seq_length=1024):
    features = []
    if file_in == "":
        return None
    if isinstance(file_in, dict):
        data = [file_in]
    elif isinstance(file_in, list):
        data = file_in
    else:
        path = Path(file_in)
        if path.is_dir():
            data = (json.loads(p.read_text()) for p in sorted(path.glob('*.json')))
        else:
            with path.open() as fh:
                first = next((c for c in iter(lambda: fh.read(1), '') if not c.isspace()), '')
            if first == '[':
                data = read_documents(path)
            else:
                data = [json.loads(path.read_text())]

    for sample in tqdm(data, desc="Example"):
        sents = []
        sent_map = []

        independent = 'spans' in sample
        source_document = sample if independent else None
        reviewed_pairs = set()
        if independent:
            text = sample['text']
            tokens = list(re.finditer(r"\w+|[^\w\s]", text))
            starts = {match.start(): index for index, match in enumerate(tokens)}
            ends = {match.end(): index + 1 for index, match in enumerate(tokens)}
            entities, identifiers = [], {}
            for span in sample['spans']:
                identifier = span['id']
                if not isinstance(identifier, str) or identifier in identifiers:
                    raise ValueError('Span IDs must be unique strings')
                start, end = span['start'], span['end']
                if (type(start) is not int or type(end) is not int or
                        not 0 <= start < end <= len(text) or
                        text[start:end] != span['text']):
                    raise ValueError('Span text and complete-token character offsets must agree')
                token_start, token_end = start, end
                while token_start < token_end and text[token_start].isspace():
                    token_start += 1
                while token_end > token_start and text[token_end - 1].isspace():
                    token_end -= 1
                if token_start not in starts or token_end not in ends or token_start == token_end:
                    raise ValueError('Span must cover complete nonempty tokens')
                if span['kind'] != 'entity':
                    raise ValueError('ATLOP expects entity spans')
                identifiers[identifier] = len(entities)
                entities.append([{'name': span['text'], 'sent_id': 0,
                                  'pos': [starts[token_start], ends[token_end]], 'type': span['label']}])
            if len(entities) < 2:
                raise ValueError('At least two entity spans are required')
            labels = []
            for relation in sample.get('relations', []):
                if relation['head'] not in identifiers or relation['tail'] not in identifiers:
                    raise ValueError('Relation references an unknown span ID')
                if relation['head'] == relation['tail'] or relation['label'] not in docred_rel2id:
                    raise ValueError('Invalid relation label or self-relation')
                labels.append({'h': identifiers[relation['head']], 't': identifiers[relation['tail']],
                               'r': relation['label']})
            complete = sample.get('relations_complete', False)
            if type(complete) is not bool:
                raise ValueError('relations_complete must be a boolean')
            if complete:
                reviewed_pairs = {(h, t) for h in range(len(entities))
                                  for t in range(len(entities)) if h != t}
            for pair in sample.get('reviewed_pairs', []):
                if pair['head'] not in identifiers or pair['tail'] not in identifiers or pair['head'] == pair['tail']:
                    raise ValueError('Invalid reviewed pair endpoints')
                reviewed_pairs.add((identifiers[pair['head']], identifiers[pair['tail']]))
            sample = {'title': sample['document_id'], 'sents': [[m.group() for m in tokens]],
                      'vertexSet': entities, 'labels': labels}
        else:
            entities = sample['vertexSet']
        entity_start, entity_end = [], []
        for entity in entities:
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]
                entity_start.append((sent_id, pos[0],))
                entity_end.append((sent_id, pos[1] - 1,))
        for i_s, sent in enumerate(sample['sents']):
            new_map = {}
            for i_t, token in enumerate(sent):
                tokens_wordpiece = tokenizer.tokenize(token)
                if (i_s, i_t) in entity_start:
                    tokens_wordpiece = ["*"] + tokens_wordpiece
                if (i_s, i_t) in entity_end:
                    tokens_wordpiece = tokens_wordpiece + ["*"]
                new_map[i_t] = len(sents)
                sents.extend(tokens_wordpiece)
            new_map[i_t + 1] = len(sents)
            sent_map.append(new_map)

        train_triple = {}
        if "labels" in sample:
            for label in sample['labels']:
                r = int(docred_rel2id[label['r']])
                if (label['h'], label['t']) not in train_triple:
                    train_triple[(label['h'], label['t'])] = [r]
                else:
                    train_triple[(label['h'], label['t'])].append(r)

        entity_pos = []
        for e in entities:
            entity_pos.append([])
            for m in e:
                start = sent_map[m["sent_id"]][m["pos"][0]]
                end = sent_map[m["sent_id"]][m["pos"][1]]
                entity_pos[-1].append((start, end,))

        relations, hts = [], []
        for h, t in train_triple.keys():
            relation = [-1 if independent and (h, t) not in reviewed_pairs else 0] * len(docred_rel2id)
            for relation_id in train_triple[h, t]:
                relation[relation_id] = 1
            if independent and relation[0] == 1:
                if any(value == 1 for value in relation[1:]):
                    raise ValueError('No-relation and positive labels conflict')
                relation = [1] + [0] * (len(docred_rel2id) - 1)
            relations.append(relation)
            hts.append([h, t])

        for h in range(len(entities)):
            for t in range(len(entities)):
                if h != t and (h, t) not in train_triple:
                    relation = ([-1] * len(docred_rel2id) if independent and (h, t) not in reviewed_pairs else
                                [1] + [0] * (len(docred_rel2id) - 1))
                    relations.append(relation)
                    hts.append([h, t])

        assert len(relations) == len(entities) * (len(entities) - 1)

        if independent and any(end > max_seq_length - 2 for mentions in entity_pos for start, end in mentions):
            raise ValueError('A selected span exceeds the model token limit')
        sents = sents[:max_seq_length - 2]
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        feature = {'input_ids': input_ids,
                   'entity_pos': entity_pos,
                   'labels': relations,
                   'hts': hts,
                   'title': sample['title'],
                   'format': 'spans' if independent else 'docred',
                   }
        if independent:
            feature['document'] = source_document
            feature['span_ids'] = [span['id'] for span in source_document['spans']]
        features.append(feature)

    return features
