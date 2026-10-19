import torch
import torch.nn.functional as F
import numpy as np


def process_long_input(model, input_ids, attention_mask, start_tokens, end_tokens):
    # Split the input to 2 overlapping chunks. Now BERT can encode inputs of which the length are up to 1024.
    n, c = input_ids.size()
    start_tokens = torch.tensor(start_tokens).to(input_ids)
    end_tokens = torch.tensor(end_tokens).to(input_ids)
    len_start = start_tokens.size(0)
    len_end = end_tokens.size(0)
    if c > 1024 - len_start - len_end:
        return process_windows(model, input_ids, attention_mask, start_tokens, end_tokens)
    if c <= 512:
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        sequence_output = output[0]
        attention = output[-1][-1]
    else:
        new_input_ids, new_attention_mask, num_seg = [], [], []
        seq_len = attention_mask.sum(1).cpu().numpy().astype(np.int32).tolist()
        for i, l_i in enumerate(seq_len):
            if l_i <= 512:
                new_input_ids.append(input_ids[i, :512])
                new_attention_mask.append(attention_mask[i, :512])
                num_seg.append(1)
            else:
                input_ids1 = torch.cat([input_ids[i, :512 - len_end], end_tokens], dim=-1)
                input_ids2 = torch.cat([start_tokens, input_ids[i, (l_i - 512 + len_start): l_i]], dim=-1)
                attention_mask1 = attention_mask[i, :512]
                attention_mask2 = attention_mask[i, (l_i - 512): l_i]
                new_input_ids.extend([input_ids1, input_ids2])
                new_attention_mask.extend([attention_mask1, attention_mask2])
                num_seg.append(2)
        input_ids = torch.stack(new_input_ids, dim=0)
        attention_mask = torch.stack(new_attention_mask, dim=0)
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        sequence_output = output[0]
        attention = output[-1][-1]
        i = 0
        new_output, new_attention = [], []
        for (n_s, l_i) in zip(num_seg, seq_len):
            if n_s == 1:
                output = F.pad(sequence_output[i], (0, 0, 0, c - 512))
                att = F.pad(attention[i], (0, c - 512, 0, c - 512))
                new_output.append(output)
                new_attention.append(att)
            elif n_s == 2:
                output1 = sequence_output[i][:512 - len_end]
                mask1 = attention_mask[i][:512 - len_end]
                att1 = attention[i][:, :512 - len_end, :512 - len_end]
                output1 = F.pad(output1, (0, 0, 0, c - 512 + len_end))
                mask1 = F.pad(mask1, (0, c - 512 + len_end))
                att1 = F.pad(att1, (0, c - 512 + len_end, 0, c - 512 + len_end))

                output2 = sequence_output[i + 1][len_start:]
                mask2 = attention_mask[i + 1][len_start:]
                att2 = attention[i + 1][:, len_start:, len_start:]
                output2 = F.pad(output2, (0, 0, l_i - 512 + len_start, c - l_i))
                mask2 = F.pad(mask2, (l_i - 512 + len_start, c - l_i))
                att2 = F.pad(att2, [l_i - 512 + len_start, c - l_i, l_i - 512 + len_start, c - l_i])
                mask = mask1 + mask2 + 1e-10
                output = (output1 + output2) / mask.unsqueeze(-1)
                att = (att1 + att2)
                att = att / (att.sum(-1, keepdim=True) + 1e-10)
                new_output.append(output)
                new_attention.append(att)
            i += n_s
        sequence_output = torch.stack(new_output, dim=0)
        attention = torch.stack(new_attention, dim=0)
    return sequence_output, attention


def process_windows(model, input_ids, attention_mask, start_tokens, end_tokens):
    """Cover the middle of documents that cannot fit into two BERT windows."""
    width = input_ids.size(1)
    capacity = 512 - len(start_tokens) - len(end_tokens)
    outputs, attentions = [], []
    for ids, mask in zip(input_ids, attention_mask):
        length = int(mask.sum().item())
        pieces, attention_pieces, coverage = [], [], []
        for start in range(0, length, capacity // 2):
            end = min(start + capacity, length)
            prefix = start_tokens if start else start_tokens[:0]
            suffix = end_tokens if end < length else end_tokens[:0]
            chunk = torch.cat([prefix, ids[start:end], suffix]).unsqueeze(0)
            encoded = model(input_ids=chunk, attention_mask=torch.ones_like(chunk),
                            output_attentions=True)
            left, right = len(prefix), len(prefix) + end - start
            pieces.append(F.pad(encoded[0][0, left:right], (0, 0, start, width - end)))
            attention_pieces.append(F.pad(encoded[-1][-1][0, :, left:right, left:right],
                                         (start, width - end, start, width - end)))
            coverage.append(F.pad(torch.ones(end - start, device=ids.device),
                                  (start, width - end)))
        count = torch.stack(coverage).sum(0).clamp_min(1)
        outputs.append(torch.stack(pieces).sum(0) / count.unsqueeze(-1))
        attention = torch.stack(attention_pieces).sum(0)
        attentions.append(attention / attention.sum(-1, keepdim=True).clamp_min(1e-10))
    return torch.stack(outputs), torch.stack(attentions)
