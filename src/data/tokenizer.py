"""Byte-level tokenizer for text encoding."""

import torch
from typing import Tuple, List

from transformers import T5Tokenizer


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

class HFTokenizerWrapper:
    """
    HuggingFace tokenizer wrapper (e.g., T5Tokenizer).
    Interface-compatible with ByteLevelTokenizer for collators.
    """

    def __init__(
        self,
        model_name: str = "t5-base",
        max_len: int = 256,
        padding: str = "max_length",  # or "longest"
    ):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.max_len = max_len
        self.padding = padding

    def encode_batch(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            texts: list[str]

        Returns:
            input_ids: [B, L]
            attention_mask: [B, L]
        """
        enc = self.tokenizer(
            texts,
            padding=self.padding,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return enc["input_ids"].long(), enc["attention_mask"].long()

    def encode(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        (Optional) single-text encode, for API compatibility
        """
        enc = self.tokenizer(
            text,
            padding=self.padding,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return enc["input_ids"][0].long(), enc["attention_mask"][0].long()