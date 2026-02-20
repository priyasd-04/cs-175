from __future__ import annotations

import torch
import torch.nn as nn


class TransformerJudge(nn.Module):
    """Binary classifier on top of a frozen TransformerMonkey.

    The base transformer is treated as a feature extractor; we attach a small
    MLP head that predicts Shakespeare-likeliness in [0, 1].
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.transformer = base_model
        # Freeze base model weights.
        for param in self.transformer.parameters():
            param.requires_grad = False

        # Ensure deterministic base representations (disable dropout, etc.).
        self.transformer.eval()

        self.classifier = nn.Sequential(
            nn.Linear(base_model.lm_head.in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),  # logits
        )

    def forward(
        self,
        idx: torch.Tensor,
        *,
        lengths: torch.Tensor | None = None,
        pad_id: int = 0,
    ) -> torch.Tensor:
        """Return Shakespeare-likeliness for each sequence.

        Args:
            idx: LongTensor of shape (B, T) with token ids.
            lengths: Optional LongTensor of shape (B,) giving the number of
                non-padding tokens per row. If provided, we take the hidden
                state at lengths-1. If omitted, we use the last position (T-1).
            pad_id: Token id used for padding (only relevant when lengths is provided).
        """
        with torch.no_grad():
            token_embeddings = self.transformer.token_embedding_table(idx)
            pos_embeddings = self.transformer.position_embedding_table(torch.arange(idx.shape[1], device=idx.device))
            x = token_embeddings + pos_embeddings

            # Create causal mask.
            T = idx.shape[1]
            # Bool mask: True means "mask out" (do not attend).
            mask = torch.triu(torch.ones((T, T), device=idx.device, dtype=torch.bool), diagonal=1)

            # Mask padding tokens if lengths were provided.
            key_padding_mask = None
            if lengths is not None:
                key_padding_mask = idx.eq(int(pad_id))

            x = self.transformer.blocks(x, mask=mask, src_key_padding_mask=key_padding_mask, is_causal=True)
            x = self.transformer.ln_f(x)

        if lengths is None:
            last_hidden = x[:, -1, :]
        else:
            # Clamp lengths to [1, T] and gather the last non-pad hidden state.
            T = x.shape[1]
            lengths = torch.clamp(lengths.to(x.device), 1, T)
            gather_idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, x.shape[2])
            last_hidden = x.gather(dim=1, index=gather_idx).squeeze(1)

        logits = self.classifier(last_hidden)
        return torch.sigmoid(logits)