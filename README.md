# learning-ai-components

Understand AI mechanisms through code, training, evaluation, and working applications.

This repository contains learning experiments for studying AI components, including language models and information extraction. Implementations may simplify or adapt the original code, datasets, models, and settings, so results may differ from the original work. The goal is to understand each mechanism, not to fully reproduce published model experiments or benchmark results.

- **Learn** — `<cat>_<mechanism>/main.ipynb`. The code is the explanation.
- **Test** — copy [questions/template.md](questions/template.md), answer with the notebook closed.

New module: [AGENTS.md](AGENTS.md).

## Dependency graph

Category flow, left to right: tokenization → architecture → training → post-training → inference → agent. Numbers match the [index](#index) learning order. Other experiments sit outside this flow. Node color = category; the training block folds in the foundations primitives (blue).

```mermaid
flowchart LR
    subgraph TOKENIZE["① tokenization"]
        direction TB
        M1[1 · BPE tokenizer]:::token
    end

    subgraph ARCH["② architecture"]
        direction TB
        M6[6 · embeddings + RoPE]:::arch
        M8[8 · LayerNorm & RMSNorm]:::arch
        M7[7 · attention]:::arch
        M9[9 · full GPT]:::arch
        M6 --> M7 --> M9
        M8 --> M9
    end

    subgraph TRAIN["③ training"]
        direction TB
        M2[2 · autograd]:::found
        M3[3 · cross-entropy]:::found
        M4[4 · training loop + SGD]:::train
        M5[5 · AdamW]:::found
        M12[12 · grad accum / clip / LR]:::train
        M2 --> M3 --> M4 --> M5 --> M12
    end

    subgraph POST["④ post-training"]
        direction TB
        M13[13 · SFT]:::post
        M14[14 · DPO]:::post
        M15[15 · GRPO]:::post
        M19[19 · tool-calling SFT]:::post
        M22[22 · Agentic Self-Instruct]:::post
        M13 --> M14
        M13 --> M15
        M13 --> M19
        M13 --> M22
    end

    subgraph INFER["⑤ inference"]
        direction TB
        M10[10 · sampling]:::infer
        M24[24 · structured output]:::infer
        M10 --> M24
        M11[11 · KV-cache]:::infer
    end

    subgraph AGENT["⑥ agent"]
        direction TB
        M16[16 · tool calling]:::agent
        M17[17 · ReAct loop]:::agent
        M18[18 · multi-step + memory]:::agent
        M20[20 · autoresearch]:::agent
        M21[21 · meta-evolution]:::agent
        M16 --> M17 --> M18 --> M20 --> M21
    end

    subgraph OTHER["other"]
        M23[23 · DiZiNER]:::other
        M25[25 · entity linking]:::other
    end

    %% block-to-block flow
    TOKENIZE ==> ARCH ==> TRAIN ==> POST ==> INFER ==> AGENT

    classDef found fill:#e3f2fd,stroke:#1565c0,color:#0d1b2a
    classDef train fill:#fff3e0,stroke:#e65100,color:#0d1b2a
    classDef token fill:#f3e5f5,stroke:#6a1b9a,color:#0d1b2a
    classDef arch fill:#e8f5e9,stroke:#2e7d32,color:#0d1b2a
    classDef infer fill:#fce4ec,stroke:#c2185b,color:#0d1b2a
    classDef post fill:#ede7f6,stroke:#4527a0,color:#0d1b2a
    subgraph IE["ie · information extraction"]
        M26[26 · ATLOP baseline]:::ie
    end
    classDef ie fill:#eeeeee,stroke:#616161,color:#0d1b2a
    classDef other fill:#eeeeee,stroke:#616161,color:#0d1b2a
    classDef agent fill:#e0f2f1,stroke:#00695c,color:#0d1b2a
    style TOKENIZE fill:#fafafa,stroke:#bbb,color:#333
    style ARCH fill:#fafafa,stroke:#bbb,color:#333
    style TRAIN fill:#fafafa,stroke:#bbb,color:#333
    style POST fill:#fafafa,stroke:#bbb,color:#333
    style INFER fill:#fafafa,stroke:#bbb,color:#333
    style OTHER fill:#fafafa,stroke:#bbb,color:#333
    style AGENT fill:#fafafa,stroke:#bbb,color:#333
```

## Index

23 core mechanisms, 1 information extraction baseline, and 2 other experiments, grouped by category in the [graph](#dependency-graph)'s order. The **#** is the learning order — follow it, or take a category at a time. Structured output (#24) can be studied immediately after sampling (#10).

Dirs are `<cat>_<mechanism>` — `tok` · `fnd` · `trn` · `arc` · `inf` · `pst` · `agt` · `ie` · `other`. Both
halves are stable, so **this table owns the order**: adding or dropping a mechanism is a
one-row edit and never renames a directory.

| # | Mechanism | Category | Status | Verified against | One-line takeaway |
|---|-----------|----------|--------|------------------|-------------------|
| 1 | [BPE tokenizer](tok_bpe/) | tokenization | 🔲 | HF `tokenizers` | — |
| 6 | [embeddings + RoPE](arc_rope/) | architecture | 🔲 | Llama reference impl | — |
| 7 | [attention (single → multi-head)](arc_attention/) | architecture | 🔲 | `F.scaled_dot_product_attention` | — |
| 8 | [LayerNorm & RMSNorm](arc_layernorm/) | architecture | 🔲 | `nn.LayerNorm` / Llama RMSNorm | — |
| 9 | [full GPT](arc_gpt/) | architecture | 🔲 | nanoGPT | — |
| 2 | [autograd](fnd_autograd/) | foundations | 🔲 | `torch.autograd` | — |
| 3 | [cross-entropy](fnd_cross_entropy/) | foundations | 🔲 | `F.cross_entropy` | — |
| 5 | [AdamW](fnd_adamw/) | foundations | 🔲 | `torch.optim.AdamW` | — |
| 4 | [training loop + SGD](trn_loop/) | training | 🔲 | `torch.optim.SGD` | — |
| 12 | [grad accumulation + clipping + LR schedules](trn_grad_accumulation/) | training | 🔲 | math identity: N steps ≡ batch×N | — |
| 13 | [SFT with prompt masking](pst_sft/) | post-training | 🔲 | `trl.SFTTrainer` | — |
| 14 | [DPO](pst_dpo/) | post-training | 🔲 | `trl.DPOTrainer` | — |
| 15 | [GRPO](pst_grpo/) | post-training | 🔲 | `trl` / `verl` | — |
| 19 | [tool-calling SFT](pst_tool_sft/) | post-training | 🟡 | `Qwen2.5` chat template (rendered) | `tools=` is a prompt + a fine-tuned habit + a parser — none of it enforced |
| 22 | [Agentic Self-Instruct](pst_autodata/) | post-training | 🟡 | Autodata ([arXiv:2606.25996](https://arxiv.org/abs/2606.25996)) — gap `0.02 → 0.314` | An example only teaches if the strong model gets it right and the weak one doesn't |
| 10 | [sampling (greedy/temp/top-k/top-p)](inf_sampling/) | inference | 🔲 | HF `generate` | — |
| 24 | [structured output (constrained decoding)](inf_structured_output/) | inference | 🟡 | Offline masking + JSON Schema checks; llama.cpp b10883 live comparison | Constrained decoding restricts what the model can emit; it does not guarantee the content is correct |
| 11 | [KV-cache](inf_kv_cache/) | inference | 🔲 | no-cache generation (identical outputs) | — |
| 16 | [tool calling](agt_tool_calling/) | agent | 🟡 | OpenAI tool-call wire format (DeepInfra) | Model returns structured `tool_calls`; you execute and feed results back |
| 17 | [ReAct loop](agt_react_loop/) | agent | 🟡 | `ysymyth/ReAct` | Interleave Thought → Action → Observation until Finish[answer] |
| 18 | [multi-step + memory](agt_memory/) | agent | 🟡 | LangGraph reference loop | The API is stateless; memory is whatever you choose to re-send |
| 20 | [autoresearch](agt_autoresearch/) | agent | 🟡 | [karpathy/autoresearch](https://github.com/karpathy/autoresearch) (one loop, `program.md`) | Edit one file, measure, keep or discard — the human is the bottleneck, so remove them |
| 21 | [meta-evolution](agt_meta_evolution/) | agent | 🟡 | EvoX ([arXiv:2602.23413](https://arxiv.org/abs/2602.23413)) · [skydiscover](https://github.com/skydiscover-ai/skydiscover) | The search strategy is code — a second loop rewrites it when `Δ < τ` says the first one stalled |
| 23 | [DiZiNER: disagreement-guided NER](other_diziner/) | other | 🟡 | [ACL 2026 paper](https://aclanthology.org/2026.acl-long.795/) · Thai/English adaptation, strict span F1 | Independent annotators disagree; a supervisor refines their instructions without gold labels or weight updates |
| 25 | [entity linking](other_entity_linking/) | other | 🟡 | ReFinED at `7f57bd2` · pretrained CPU inference | Context, types, descriptions, and priors resolve an ambiguous mention to an entity |
| 26 | [ATLOP baseline](ie_atlop/) | ie | 🟡 | Official ATLOP `1db77ab8`; dev F1 60.80 / Ign F1 58.97 (seed 66, epoch 29) | Each entity pair learns a threshold and pools context attended to by both entities |

Status legend: 🔲 planned · 🟡 in progress · ✅ done (matches the reference). Agent modules score by EM/F1 or exact value, not allclose.

## Setup

Modules are separate projects: one venv each, from the module's own `pyproject.toml`. The `.env`
is the exception — one at the root, shared.

```bash
cp .env.example .env    # once per clone: add DEEPINFRA_API_KEY or OPENROUTER_API_KEY for DiZiNER
cd agt_react_loop       # or any module
deactivate              # only if a root .venv is active — see below
uv sync                 # per-module venv
```

Sync from **inside** the module, never the root. If `VIRTUAL_ENV` points at a root `.venv`,
`uv sync` installs there instead and strips the other modules' deps — it removed `requests` and
broke `agt_react_loop`. Notebooks find the key by walking up: `load_dotenv(find_dotenv(usecwd=True))`.

Docker available if needed. Data, checkpoints, and logs live on `/workspace` (uncommitted).

For DiZiNER training, prompt inspection, and matched testing, see the [module run instructions](other_diziner/README.md#run).
