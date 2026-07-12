"""Flat-cost fake model for mechanism runs.

The modeling assumption, load-bearing and chosen: a step costs delay_s
regardless of batch composition, so batching is free. That is the GPU serving
regime (weight reads amortize across the batch), and the opposite of this
repo's per-sequence CPU forward. Both benchmark arms enjoy it equally, so the
comparison stays fair; the mechanism chart answers "how do these policies
compare when batching is free." Prefill likewise costs one flat step
regardless of prompt length.

Never emits EOS, so the max_tokens cap IS the realized output length: the
independent variable is controlled, not observed.
"""

import time

import torch


class FlatCostModel:
    def __init__(self, delay_s: float = 0.02, vocab: int = 16):
        self.delay_s = delay_s
        self.vocab = vocab

    def forward(self, seqs, kv):
        time.sleep(self.delay_s)
        out = {}
        for s in seqs:
            v = torch.zeros(self.vocab)
            v[(len(s.output_ids) % 5) + 1] = 1.0  # deterministic, never EOS (0)
            out[s.req_id] = v
        return out
