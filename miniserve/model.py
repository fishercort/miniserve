"""Load HF weights and run the forward pass with paged KV. See docs/phase1-engine.md.

v1 attention is gather-based: each sequence's K/V blocks are gathered into a
contiguous buffer, then standard attention runs over it. Simpler and
pure-PyTorch; the perf gap vs a real paged kernel is a known, flagged tradeoff
— slow-and-identical is this module's entire contract.

The forward is written from scratch against the paged cache (RMSNorm, RoPE,
GQA, SwiGLU); HF's model object is used only as a weight container, never for
its forward — so there is no checkpoint-name mapping to silently drop a
projection, and Qwen2's Q/K/V biases come along by construction. Correctness
is pinned by tests comparing logits and greedy tokens against the HF reference.

RoPE note: positions are absolute content positions, tracked explicitly. A
resumed prefill restarts from position 0 over prompt + generated tokens — which
is CORRECT in this engine, because preemption frees all KV; nothing rotated
under the old positions survives to disagree.
"""

import math

import torch
import torch.nn.functional as F

from miniserve.kv_cache import PagedKVCache
from miniserve.scheduler import Sequence, Status


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class PagedModel:
    """Qwen2-family forward pass over a PagedKVCache.

    Processes each sequence independently (tractable-but-correct v1): prefill
    writes KV for every content position and returns last-position logits;
    decode writes KV for the newest token and attends over the gathered past.
    """

    def __init__(self, hf_model, device: str = "cpu") -> None:
        cfg = hf_model.config
        self.device = device
        self.num_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        # Prefer a declared head_dim; derive only when absent (MECHANICS.md
        # records the derivation trap: explicit fields can disagree with it).
        self.head_dim = (
            getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        )
        self.n_rep = self.n_heads // self.n_kv_heads  # GQA: Q heads per KV head
        self.rms_eps = cfg.rms_norm_eps

        m = hf_model.model
        self.embed = m.embed_tokens.weight
        self.layers = []
        for layer in m.layers:
            a, p = layer.self_attn, layer.mlp
            self.layers.append(
                {
                    "in_norm": layer.input_layernorm.weight,
                    "q_w": a.q_proj.weight, "q_b": a.q_proj.bias,  # Qwen2: QKV biased
                    "k_w": a.k_proj.weight, "k_b": a.k_proj.bias,
                    "v_w": a.v_proj.weight, "v_b": a.v_proj.bias,
                    "o_w": a.o_proj.weight,
                    "post_norm": layer.post_attention_layernorm.weight,
                    "gate_w": p.gate_proj.weight, "up_w": p.up_proj.weight,
                    "down_w": p.down_proj.weight,
                }
            )
        self.final_norm = m.norm.weight
        self.lm_head = hf_model.lm_head.weight
        if cfg.tie_word_embeddings and self.lm_head.data_ptr() != self.embed.data_ptr():
            # Enforced, not assumed: if transformers failed to materialize the
            # tie on load, lm_head is stale garbage and every logit is silently
            # wrong. A real exception so the check survives python -O.
            raise RuntimeError(
                "config declares tied embeddings but lm_head and embed_tokens "
                "are different tensors — the tie did not materialize on load"
            )
        # transformers v5 moved rope_theta into a rope_parameters dict; v4 had
        # it as a top-level attribute. Handle both, and refuse rope variants
        # (yarn, linear scaling, ...) that this default-RoPE forward would
        # silently get wrong.
        rope_params = getattr(cfg, "rope_parameters", None)
        if rope_params is not None:
            if rope_params.get("rope_type", "default") != "default":
                raise RuntimeError(
                    f"unsupported rope_type {rope_params['rope_type']!r}: "
                    "PagedModel implements default RoPE only"
                )
            if "rope_theta" not in rope_params:
                # Shims fire on environments nobody tested — fail with a shape
                # diagnosis, not a KeyError.
                raise RuntimeError(
                    f"unrecognized rope_parameters shape (keys: "
                    f"{sorted(rope_params)}): expected a rope_theta entry"
                )
            rope_theta = rope_params["rope_theta"]
        else:
            rope_theta = cfg.rope_theta
        inv_freq = 1.0 / (
            rope_theta
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        self.inv_freq = inv_freq.to(device)

    @classmethod
    def from_pretrained(cls, name_or_path: str, device: str = "cpu", dtype=None):
        from transformers import AutoModelForCausalLM

        hf = (
            AutoModelForCausalLM.from_pretrained(
                name_or_path,
                torch_dtype=dtype or torch.bfloat16,
                attn_implementation="eager",
            )
            .to(device)
            .eval()
        )
        return cls(hf, device=device)

    # -- engine-facing API -----------------------------------------------------

    @torch.inference_mode()
    def forward(
        self, seqs: list[Sequence], kv: PagedKVCache
    ) -> dict[str, torch.Tensor]:
        """Next-token logits per request. Prefill covers ALL content (prompt +
        any tokens generated before a preemption); decode covers exactly the
        newest token, whose KV slot the scheduler's growth phase guaranteed."""
        out: dict[str, torch.Tensor] = {}
        for s in seqs:
            content = s.prompt_ids + s.output_ids
            if s.status is Status.PREFILL:
                positions = torch.arange(len(content), device=self.device)
                tokens = torch.tensor(content, device=self.device)
            else:
                positions = torch.tensor([len(content) - 1], device=self.device)
                tokens = torch.tensor([content[-1]], device=self.device)
            out[s.req_id] = self._forward_one(s.req_id, tokens, positions, kv)
        return out

    # -- internals --------------------------------------------------------------

    def _rms(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.rms_eps)
        return (x.float() * scale).to(x.dtype) * weight

    def _rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        freqs = positions.float()[:, None] * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[:, None, :].to(x.dtype)
        sin = emb.sin()[:, None, :].to(x.dtype)
        return x * cos + _rotate_half(x) * sin

    def _forward_one(
        self,
        req_id: str,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        kv: PagedKVCache,
    ) -> torch.Tensor:
        table = kv.block_tables[req_id]
        bs = kv.block_size
        block_ids = torch.tensor(table, device=self.device)[positions // bs]
        offsets = positions % bs
        seen = int(positions[-1].item()) + 1  # keys visible: positions 0..seen-1

        x = self.embed[tokens]
        for li, w in enumerate(self.layers):
            h = self._rms(x, w["in_norm"])
            t = h.shape[0]
            q = F.linear(h, w["q_w"], w["q_b"]).view(t, self.n_heads, self.head_dim)
            k = F.linear(h, w["k_w"], w["k_b"]).view(t, self.n_kv_heads, self.head_dim)
            v = F.linear(h, w["v_w"], w["v_b"]).view(t, self.n_kv_heads, self.head_dim)
            q, k = self._rope(q, positions), self._rope(k, positions)

            # Write this pass's K/V into the paged blocks (rotated at write
            # time, under absolute positions)...
            kv.k_cache[li][block_ids, offsets] = k
            kv.v_cache[li][block_ids, offsets] = v
            # ...then gather the sequence's whole past into a contiguous buffer.
            k_all = kv.k_cache[li][table].reshape(-1, self.n_kv_heads, self.head_dim)
            v_all = kv.v_cache[li][table].reshape(-1, self.n_kv_heads, self.head_dim)
            k_all = k_all[:seen].repeat_interleave(self.n_rep, dim=1)  # GQA share
            v_all = v_all[:seen].repeat_interleave(self.n_rep, dim=1)

            scores = torch.einsum("thd,shd->hts", q, k_all) / math.sqrt(self.head_dim)
            key_pos = torch.arange(seen, device=self.device)
            causal = key_pos[None, :] <= positions[:, None]  # [t, seen]
            scores = scores.masked_fill(~causal[None], torch.finfo(scores.dtype).min)
            attn = torch.softmax(scores.float(), dim=-1).to(x.dtype)
            ctx = torch.einsum("hts,shd->thd", attn, v_all).reshape(t, -1)
            x = x + F.linear(ctx, w["o_w"])

            h = self._rms(x, w["post_norm"])
            gate = F.silu(F.linear(h, w["gate_w"])) * F.linear(h, w["up_w"])
            x = x + F.linear(gate, w["down_w"])

        h_last = self._rms(x[-1], self.final_norm)
        return F.linear(h_last, self.lm_head)
