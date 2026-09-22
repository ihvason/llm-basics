"""Tokenization layer: the BPE tokenizer and GPT-2 style pre-tokenization.

- ``tokenizer.BPETokenizer``: training, encoding/decoding and serialization.
- ``pretokenization``: splitting raw text into candidate word chunks.
"""

from .tokenizer import BPETokenizer

__all__ = ["BPETokenizer"]
