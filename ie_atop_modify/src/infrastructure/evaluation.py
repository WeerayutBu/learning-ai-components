import os
from pathlib import Path
import json
import numpy as np

from .prepro import docred_rel2id as rel2id
id2rel = {value: key for key, value in rel2id.items()}


def to_official(preds, features):
    h_idx, t_idx, title = [], [], []

    for f in features:
        hts = f["hts"]
        h_idx += [ht[0] for ht in hts]
        t_idx += [ht[1] for ht in hts]
        title += [f["title"] for ht in hts]

    res = []
    for i in range(preds.shape[0]):
        pred = preds[i]
        pred = np.nonzero(pred)[0].tolist()
        for p in pred:
            if p != 0:
                res.append(
                    {
                        'title': title[i],
                        'h_idx': h_idx[i],
                        't_idx': t_idx[i],
                        'r': id2rel[p],
                    }
                )
    return res


def gen_train_facts(data_file_name, truth_dir):
    fact_file_name = data_file_name[data_file_name.find("train_"):]
    fact_file_name = os.path.join(truth_dir, fact_file_name.replace(".json", ".fact"))

    if os.path.exists(fact_file_name):
        fact_in_train = set([])
        triples = json.loads(Path(fact_file_name).read_text())
        for x in triples:
            fact_in_train.add(tuple(x))
        return fact_in_train

    fact_in_train = set([])
    ori_data = json.loads(Path(data_file_name).read_text())
    for data in ori_data:
        vertexSet = data['vertexSet']
        for label in data['labels']:
            rel = label['r']
            for n1 in vertexSet[label['h']]:
                for n2 in vertexSet[label['t']]:
                    fact_in_train.add((n1['name'], n2['name'], rel))

    Path(fact_file_name).write_text(json.dumps(list(fact_in_train)))

    return fact_in_train


def official_evaluate(tmp, path, dev_file="dev.json"):
    '''
        Adapted from the official evaluation code
    '''
    if not tmp:
        return 0.0, 0.0, 0.0, 0.0
    truth_dir = os.path.join(path, 'ref')

    if not os.path.exists(truth_dir):
        os.makedirs(truth_dir)

    fact_in_train_annotated = gen_train_facts(os.path.join(path, "train_annotated.json"), truth_dir)
    fact_in_train_distant = gen_train_facts(os.path.join(path, "train_distant.json"), truth_dir)

    truth = json.loads((Path(path) / dev_file).read_text())

    std = {}
    tot_evidences = 0

    title2vectexSet = {}

    for x in truth:
        title = x['title']

        vertexSet = x['vertexSet']
        title2vectexSet[title] = vertexSet

        for label in x['labels']:
            r = label['r']
            h_idx = label['h']
            t_idx = label['t']
            std[(title, r, h_idx, t_idx)] = set(label['evidence'])
            tot_evidences += len(label['evidence'])

    tot_relations = len(std)
    tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
    submission_answer = [tmp[0]]
    for i in range(1, len(tmp)):
        x = tmp[i]
        y = tmp[i - 1]
        if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
            submission_answer.append(tmp[i])

    correct_re = 0
    correct_evidence = 0
    pred_evi = 0

    correct_in_train_annotated = 0
    correct_in_train_distant = 0
    for x in submission_answer:
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']
        if title not in title2vectexSet:
            continue
        vertexSet = title2vectexSet[title]

        if 'evidence' in x:
            evi = set(x['evidence'])
        else:
            evi = set([])
        pred_evi += len(evi)

        if (title, r, h_idx, t_idx) in std:
            correct_re += 1
            stdevi = std[(title, r, h_idx, t_idx)]
            correct_evidence += len(stdevi & evi)
            in_train_annotated = in_train_distant = False
            for n1 in vertexSet[h_idx]:
                for n2 in vertexSet[t_idx]:
                    if (n1['name'], n2['name'], r) in fact_in_train_annotated:
                        in_train_annotated = True
                    if (n1['name'], n2['name'], r) in fact_in_train_distant:
                        in_train_distant = True

            if in_train_annotated:
                correct_in_train_annotated += 1
            if in_train_distant:
                correct_in_train_distant += 1

    re_p = 1.0 * correct_re / len(submission_answer)
    re_r = 1.0 * correct_re / tot_relations
    if re_p + re_r == 0:
        re_f1 = 0
    else:
        re_f1 = 2.0 * re_p * re_r / (re_p + re_r)

    evi_p = 1.0 * correct_evidence / pred_evi if pred_evi > 0 else 0
    evi_r = 1.0 * correct_evidence / tot_evidences
    if evi_p + evi_r == 0:
        evi_f1 = 0
    else:
        evi_f1 = 2.0 * evi_p * evi_r / (evi_p + evi_r)

    re_p_ignore_train_annotated = 1.0 * (correct_re - correct_in_train_annotated) / (len(submission_answer) - correct_in_train_annotated + 1e-5)
    re_p_ignore_train = 1.0 * (correct_re - correct_in_train_distant) / (len(submission_answer) - correct_in_train_distant + 1e-5)

    if re_p_ignore_train_annotated + re_r == 0:
        re_f1_ignore_train_annotated = 0
    else:
        re_f1_ignore_train_annotated = 2.0 * re_p_ignore_train_annotated * re_r / (re_p_ignore_train_annotated + re_r)

    if re_p_ignore_train + re_r == 0:
        re_f1_ignore_train = 0
    else:
        re_f1_ignore_train = 2.0 * re_p_ignore_train * re_r / (re_p_ignore_train + re_r)

    return re_f1, evi_f1, re_f1_ignore_train_annotated, re_f1_ignore_train


def span_predictions(preds, features):
    """Export one self-contained document, including empty predictions, per input."""
    documents, offset = [], 0
    for feature in features:
        source = feature['document']
        relations = []
        for row, (head, tail) in zip(preds[offset:offset + len(feature['hts'])], feature['hts']):
            for label in np.flatnonzero(row):
                if label:
                    relations.append({'head': feature['span_ids'][head],
                                      'tail': feature['span_ids'][tail], 'label': id2rel[label]})
        documents.append({'document_id': source['document_id'], 'text': source['text'],
                          'spans': source['spans'], 'relations': relations})
        offset += len(feature['hts'])
    if offset != len(preds):
        raise ValueError('Prediction rows do not match input pairs')
    return documents


def evaluate_spans(documents, features):
    """Micro scores on known labels only; unknowns cannot be false positives."""
    predicted = {doc['document_id']: doc for doc in documents}
    tp = fp = fn = known = total = positives = unknown_predictions = 0
    for feature in features:
        identifiers = feature['span_ids']
        edges = {(r['head'], r['tail'], r['label'])
                 for r in predicted[feature['title']]['relations']}
        for (head, tail), labels in zip(feature['hts'], feature['labels']):
            for index, gold in enumerate(labels[1:], 1):
                total += 1
                found = (identifiers[head], identifiers[tail], id2rel[index]) in edges
                if gold < 0:
                    unknown_predictions += int(found)
                    continue
                known += 1
                positives += int(gold == 1)
                tp += int(found and gold == 1)
                fp += int(found and gold == 0)
                fn += int(not found and gold == 1)
    if not known or not positives:
        raise ValueError('Dev data needs known positive labels for checkpoint selection')
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1,
            'known_labels': known, 'total_labels': total,
            'unknown_predictions': unknown_predictions}
