"""Validate user input and pass a DocRED document to the supplied predictor."""
import re

tokenize = re.compile(r"\w+|[^\w\s]").findall


def infer(text, entities, predict):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter a document.")
    if len(text) > 12000:
        raise ValueError("Keep the document under 12,000 characters.")
    sentences = [tokenize(line) for line in text.splitlines() if line.strip()]
    vertices = []
    boundaries = []
    offset = 0
    for line in text.splitlines(keepends=True):
        if line.strip():
            boundaries.append([(offset + m.start(), offset + m.end())
                               for m in re.finditer(r"\w+|[^\w\s]", line)])
        offset += len(line)
    if isinstance(entities, list):
        names = []
        used = set()
        for entity in entities:
            if not isinstance(entity, dict) or not isinstance(entity.get("name"), str):
                raise ValueError("Each entity needs a name and selected mentions.")
            name = entity["name"].strip()
            if not name or not isinstance(entity.get("mentions"), list) or not entity["mentions"]:
                raise ValueError("Each entity needs a name and at least one mention.")
            names.append(name)
            mentions = []
            for selection in entity["mentions"]:
                if not isinstance(selection, dict):
                    raise ValueError("Invalid mention selection.")
                start, end = selection.get("start"), selection.get("end")
                if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
                    raise ValueError("Invalid mention offsets.")
                if (start, end) in used:
                    raise ValueError("A selected mention can belong to only one entity.")
                matched = None
                for sentence_id, positions in enumerate(boundaries):
                    starts = [pair[0] for pair in positions]
                    ends = [pair[1] for pair in positions]
                    if start in starts and end in ends:
                        matched = {"name": name, "sent_id": sentence_id,
                                   "pos": [starts.index(start), ends.index(end) + 1], "type": "MISC"}
                        break
                if matched is None:
                    raise ValueError("Select complete tokens within one sentence, without surrounding spaces.")
                used.add((start, end))
                mentions.append(matched)
            vertices.append(mentions)
    elif isinstance(entities, str):
        names = [name.strip() for name in entities.splitlines() if name.strip()]
        for name in names:
            tokens = tokenize(name)
            mentions = []
            for sentence_id, sentence in enumerate(sentences):
                for start in range(len(sentence) - len(tokens) + 1):
                    if sentence[start:start + len(tokens)] == tokens:
                        mentions.append({"name": name, "sent_id": sentence_id,
                                         "pos": [start, start + len(tokens)], "type": "MISC"})
            if not mentions:
                raise ValueError(f'Entity "{name}" was not found. Match its spelling and capitalization.')
            vertices.append(mentions)
    else:
        raise ValueError("Enter entity names or annotate mentions in the document.")
    if not 2 <= len(names) <= 30:
        raise ValueError("Enter between 2 and 30 entities.")
    if len(set(names)) != len(names):
        raise ValueError("Use a unique name for each entity group.")
    document = {"title": "Web input", "sents": sentences, "vertexSet": vertices}
    relations = predict(document)
    spans = {
        name: [{"sentence": mention["sent_id"], "start": mention["pos"][0],
                "end": mention["pos"][1]} for mention in mentions]
        for name, mentions in zip(names, vertices)
    }
    relations = [dict(relation, head_spans=spans[relation["head"]],
                      tail_spans=spans[relation["tail"]]) for relation in relations]
    annotations = []
    for name, mentions in zip(names, vertices):
        exported_mentions = []
        for mention in mentions:
            sentence = mention["sent_id"]
            token_start, token_end = mention["pos"]
            start = boundaries[sentence][token_start][0]
            end = boundaries[sentence][token_end - 1][1]
            exported_mentions.append({"mention": text[start:end], "start": start, "end": end,
                                      "token_span": {"sentence": sentence,
                                                     "start": token_start, "end": token_end}})
        annotations.append({"name": name, "mentions": exported_mentions})
    return {"text": text, "annotations": annotations,
            "span_convention": {"index_base": 0, "end": "exclusive",
                                "annotations": "document Unicode character offsets",
                                "token_span": "sentence-local token offsets",
                                "head_spans_and_tail_spans": "sentence-local token offsets"},
            "relations": relations, "entities": len(names),
            "pairs": len(names) * (len(names) - 1)}
