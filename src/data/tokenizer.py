"""Byte-level tokenizer for text encoding."""

import torch
from typing import Tuple


class ByteLevelTokenizer:
    """Simple byte-level tokenizer for UTF-8 text."""

    def __init__(self, pad_token: int = 0, sot_token: int = 2, eot_token: int = 3):
        self.pad_token = pad_token
        self.sot_token = sot_token
        self.eot_token = eot_token
        self.vocab_size = 256

    def encode(self, text: str, max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text to byte-level token IDs.

        Args:
            text: input text string
            max_len: maximum sequence length

        Returns:
            input_ids: [max_len] token IDs
            attention_mask: [max_len] mask (1 for valid, 0 for padding)
        """
        b = text.encode("utf-8", errors="ignore")
        ids = [self.sot_token] + list(b)[: max_len - 2] + [self.eot_token]

        # Pad to max_len
        if len(ids) < max_len:
            ids += [self.pad_token] * (max_len - len(ids))

        ids = torch.tensor(ids, dtype=torch.long)
        attention_mask = (ids != self.pad_token).long()

        return ids, attention_mask

    def decode(self, ids: torch.Tensor) -> str:
        """
        Decode token IDs back to text.

        Args:
            ids: [L] token IDs

        Returns:
            decoded text string
        """
        # Remove special tokens
        ids = ids.tolist() if torch.is_tensor(ids) else ids
        byte_ids = [i for i in ids if i not in {self.pad_token, self.sot_token, self.eot_token}]

        # Convert to bytes and decode
        try:
            return bytes(byte_ids).decode("utf-8", errors="ignore")
        except:
            return ""
