"""BiLSTM semantic-boundary detector.

Given the sentence embeddings of ONE document ``E = [e_1, ..., e_n]`` with
``e_i in R^d``, the model scores every inter-sentence boundary

    position i  (between sentence i and i+1),   i = 0 .. n-2

with the probability that a semantic topic change happens there.

Architecture (matches the project spec)::

    E (n, d)
      -> BiLSTM(hidden=H)            -> H_seq (n, 2H)
      -> for each boundary i: concat[h_i ; h_{i+1}]  (4H,)
      -> Linear(4H -> 1)            -> logit_i
      (sigmoid is applied at inference / inside the loss for stability)

The module is intentionally single-document (no padding): the spec trains
with one article per step, which sidesteps variable-length padding entirely.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BiLSTMBoundary(nn.Module):
    def __init__(
        self,
        input_dim: int = 384,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # concat of two BiLSTM states -> 2 * (2H) = 4H
        self.boundary_head = nn.Linear(4 * hidden_size, 1)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """Score the boundaries of a single document.

        Parameters
        ----------
        emb : (n, d) float tensor  -- sentence embeddings of one article.

        Returns
        -------
        (n-1,) float tensor of raw logits, one per inter-sentence boundary.
        Returns an empty (0,) tensor when the article has < 2 sentences.
        """
        if emb.dim() != 2:
            raise ValueError(f"expected (n, d) tensor, got shape {tuple(emb.shape)}")
        n = emb.size(0)
        if n < 2:
            return emb.new_zeros((0,))

        seq, _ = self.lstm(emb.unsqueeze(0))     # (1, n, 2H)
        seq = seq.squeeze(0)                      # (n, 2H)
        pair = torch.cat([seq[:-1], seq[1:]], dim=-1)  # (n-1, 4H)
        logits = self.boundary_head(pair).squeeze(-1)  # (n-1,)
        return logits

    @torch.no_grad()
    def predict_proba(self, emb: torch.Tensor) -> torch.Tensor:
        """Boundary probabilities (n-1,) for a single document."""
        return torch.sigmoid(self.forward(emb))

    @torch.no_grad()
    def predict_boundaries(self, emb: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Boolean (n-1,) mask: True where a cut should be made."""
        return self.predict_proba(emb) > threshold
