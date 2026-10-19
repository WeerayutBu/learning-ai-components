"""Convert DocRED tokens to independent character spans without coreference."""
import argparse
from pathlib import Path

import hashlib
import json


def convert_document(document, relation_policy):
    if relation_policy not in {'singletons', 'expand'}:
        raise ValueError('Choose singletons or expand for group-level relations')
    sentences = document['sents']
    text_parts, positions, offset = [], [], 0
    for sentence in sentences:
        sentence_positions = []
        for index, token in enumerate(sentence):
            if index:
                offset += 1
            sentence_positions.append((offset, offset + len(token)))
            offset += len(token)
        positions.append(sentence_positions)
        text_parts.append(' '.join(sentence))
        offset += 1
    text = '\n'.join(text_parts)
    spans, groups = [], []
    for mentions in document['vertexSet']:
        identifiers = []
        for mention in mentions:
            sentence = mention['sent_id']
            start, end = mention['pos']
            if not 0 <= sentence < len(sentences) or not 0 <= start < end <= len(sentences[sentence]):
                raise ValueError('Invalid DocRED mention positions')
            char_start = positions[sentence][start][0]
            char_end = positions[sentence][end - 1][1]
            identifier = 'e' + str(len(spans))
            spans.append({'id': identifier, 'kind': 'entity', 'label': mention['type'],
                          'start': char_start, 'end': char_end, 'text': text[char_start:char_end]})
            identifiers.append(identifier)
        groups.append(identifiers)
    relations, omitted, seen = [], 0, set()
    for relation in document.get('labels', []):
        heads, tails = groups[relation['h']], groups[relation['t']]
        if relation_policy == 'singletons' and (len(heads) != 1 or len(tails) != 1):
            omitted += 1
            continue
        for head in heads:
            for tail in tails:
                key = (head, tail, relation['r'])
                if key not in seen:
                    relations.append({'head': head, 'tail': tail, 'label': relation['r']})
                    seen.add(key)
    # These singleton pairs inherit DocRED's closed-world annotation policy.
    # Every mention remains independent; ambiguous source pairs stay unreviewed.
    reviewed_pairs = []
    if relation_policy == 'singletons' and 'labels' in document:
        for h, heads in enumerate(groups):
            for t, tails in enumerate(groups):
                if h != t and len(heads) == len(tails) == 1:
                    reviewed_pairs.append({'head': heads[0], 'tail': tails[0]})
    identifier = hashlib.sha256(json.dumps(document, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]
    return {'document_id': identifier, 'text': text, 'spans': spans, 'relations': relations,
            'reviewed_pairs': reviewed_pairs}, omitted


def read_documents(path):
    """Stream a JSON array so the distant-training split need not fit in memory."""
    decoder = json.JSONDecoder()
    with path.open(encoding='utf-8') as source:
        buffer = source.read(65536).lstrip()
        if not buffer.startswith('['):
            raise ValueError('Expected a DocRED JSON array')
        buffer = buffer[1:]
        while True:
            buffer = buffer.lstrip()
            if buffer.startswith(']'):
                return
            if buffer.startswith(','):
                buffer = buffer[1:].lstrip()
            try:
                document, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                chunk = source.read(65536)
                if not chunk:
                    raise ValueError('Incomplete DocRED JSON array')
                buffer += chunk
                continue
            yield document
            buffer = buffer[end:]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--relation-policy', choices=['singletons', 'expand'], default='singletons')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.exists():
        raise ValueError('Output already exists; choose a new path')
    array_output = args.output.suffix == '.json'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not array_output:
        args.output.mkdir()
    output = args.output.open('w', encoding='utf-8') if array_output else None
    if output:
        output.write('[\n')
    documents = omitted = relations = negatives = reviewed = 0
    for document in read_documents(args.source):
        result, skipped = convert_document(document, args.relation_policy)
        encoded = json.dumps(result, ensure_ascii=False, indent=2)
        if output:
            if documents:
                output.write(',\n')
            output.write('\n'.join('  ' + line for line in encoded.splitlines()))
        else:
            target = args.output / (result['document_id'] + '.json')
            if target.exists():
                raise ValueError('Duplicate document ID')
            target.write_text(encoded + '\n', encoding='utf-8')
        documents += 1
        relations += len(result['relations'])
        omitted += skipped
        positives = {(r['head'], r['tail']) for r in result['relations']}
        reviewed += len(result['reviewed_pairs'])
        negatives += sum((pair['head'], pair['tail']) not in positives for pair in result['reviewed_pairs'])
    if output:
        output.write('\n]\n')
        output.close()
    summary = {'documents': documents, 'positive_relations': relations,
               'reviewed_pairs': reviewed, 'generated_negative_pairs': negatives,
               'omitted_group_relations': omitted, 'relation_policy': args.relation_policy}
    args.output.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
