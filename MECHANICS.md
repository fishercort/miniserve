# MECHANICS — the metal facts (Phase 0 deliverable)

Target model, its verified configuration, and the KV cache arithmetic that
every later design decision prices against. All numbers below are derived from
the model's `config.json` as published, not from a model card or memory.

## Target model: Qwen2.5-1.5B-Instruct

Chosen per the Phase 0 criteria (1B–8B, fits one GPU, exercises GQA), with two
properties that make it a good instrument:

- **Aggressive GQA:** 12 query heads share 2 KV heads — a 6× KV compression
  that makes the cache arithmetic below vivid rather than marginal.
- **Apache-2.0.** Verified against the LICENSE file in the model repo (not the
  model-card tag; tags can go stale). The Qwen2.5 family is Apache-2.0 except
  the 3B and 72B sizes, which carry a more restrictive research license —
  1.5B stays clean for a public repo.

### Verified config (from `config.json`, 2026-07)

| Field | Value | Source |
|---|---|---|
| `num_hidden_layers` | 28 | declared |
| `num_attention_heads` | 12 | declared |
| `num_key_value_heads` | 2 | declared |
| `hidden_size` | 1536 | declared |
| `head_dim` | 128 | **derived** (1536 / 12 — no explicit field in config) |
| `torch_dtype` | bfloat16 (2 B) | declared |
| `max_position_embeddings` | 32,768 | declared |
| `vocab_size` | 151,936 | declared |
| `tie_word_embeddings` | true | declared |
| Parameters | 1.54 B (1.31 B non-embedding) | model card |

The derived-vs-declared distinction matters: some configs carry a non-standard
explicit `head_dim`, and the KV formula silently breaks if the derivation is
assumed where a declaration disagrees. This config has no explicit field, so
128 is the derivation.

## KV cache arithmetic

Formula: `num_layers × 2 (K and V) × num_kv_heads × head_dim × dtype_bytes`
per token.

| Quantity | Value | Derivation |
|---|---|---|
| Bytes per token | 28,672 B = **exactly 28 KiB** | 28 × 2 × 2 × 128 × 2 |
| Bytes per 16-token block | **448 KiB** | 28 KiB × 16 |
| One 8k sequence | **224 MiB** | 28 KiB × 8192 |
| One 32k sequence (max ctx) | **896 MiB** | 4 × above |
| Weights | 2.87 GiB | 1.54 B params × 2 B (embeddings tied; untied would add ~0.4 GiB) |
| KV budget on a 24 GiB GPU | ~19.1–19.6 GiB | 24 − 2.87 − (1.5–2.0 overhead) |
| Total blocks in budget | **~44.8–45.9k** | budget ÷ 448 KiB |
| Concurrent 8k sequences | **~87–89** | budget ÷ 224 MiB |
| Blocks per maxed 32k sequence | **2,048 ≈ 4.5% of the GPU** | 32,768 ÷ 16 |

Three notes on the numbers:

1. **28 KiB/token is exact, not rounded.** 28 × 1024 = 28,672 precisely,
   because head_dim = 128 and every other factor is a small integer. Stated
   here so the round number is not mistaken for hand-waving.
2. **The overhead band (1.5–2 GiB) is an assumption, and it contains a live
   variable.** It covers CUDA context, the logits buffer (~50 MB/step at this
   vocab and full batch), and activations — and activation memory scales with
   tokens in flight per step: a large mixed prefill+decode batch peaks
   noticeably above steady-state decode. Assumed for v1, batch-dependent,
   measured in Phase 2 (activation peak vs batch composition is a named
   measurement task there).
3. **bf16, not fp16.** The config declares `torch_dtype: bfloat16`. Same
   2 bytes per element, so no arithmetic changes — recorded for fidelity.

## The GQA punchline

The same model with vanilla MHA (12 KV heads instead of 2) would need
168 KiB/token — 1.31 GiB per 8k sequence — and the same 24 GiB GPU would hold
**14** concurrent 8k sequences instead of ~88. The 6× head-count compression
is the entire difference between a batch and a fleet.

## Why this matters for the project

~45k blocks sounds abundant. But one maxed-out 32k sequence eats 2,048 blocks
— 4.5% of the entire GPU — and twenty-two such agents erase the fleet of 88
chat sequences completely. Agentic contexts don't just grow the cache; they
change its granularity of contention.

## Still to write (Phase 0 remainder)

- Mechanical account of prefill (compute-bound) vs decode
  (memory-bandwidth-bound), from the forward-pass trace.
- Extracted scoring rules from the four papers (Continuum TTL, 2605.06472
  retired-cache, SAGA WA-LRU, IntentKV — related work only, token-level).
- What Dynamo's per-region retention already ships, and how the hint
  interface differs.
