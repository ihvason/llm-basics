"""Byte-pair encoding tokenizer with GPT-2 style pre-tokenization.

The tokenizer starts from the 256 byte values, repeatedly merges the most
frequent adjacent pair, and keeps the resulting merge list. Encoding replays
those merges in order; special tokens bypass the merge process and are matched
as whole units.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
import json
import os
from pathlib import Path
import time

import numpy as np
import regex


class BPETokenizer:
    """A byte-level BPE tokenizer.

    A tokenizer is fully described by its vocabulary (token id to byte string),
    its ordered merge list and its special tokens. Two instances built from the
    same artifacts are interchangeable, which is what makes checkpoints and
    pre-tokenized datasets reproducible.
    """

    _BASE_TOKENS = {i: bytes([i]) for i in range(256)}

    _GPT2_SPLIT_RE = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        """Build a tokenizer from an in-memory vocabulary and merge list.

        Args:
            vocab: Mapping from token id to the byte string it represents.
            merges: Ordered merge rules; order defines encoding behavior.
            special_tokens: Tokens matched as indivisible units before
                pre-tokenization, e.g. ``["<|endoftext|>"]``.
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        self.inverse_vocab: dict[bytes, int] = {v: k for k, v in vocab.items()}

        if self.special_tokens:
            # Longest first so that overlapping tokens match greedily.
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            escaped = [regex.escape(t) for t in sorted_tokens]
            self._special_re = regex.compile(f"({'|'.join(escaped)})")
            self._special_set = set(self.special_tokens)
        else:
            self._special_re = None
            self._special_set = set()

    def __len__(self) -> int:
        """Return the number of entries in the vocabulary."""
        return len(self.vocab)

    @property
    def vocab_size(self) -> int:
        """Vocabulary size."""
        return len(self.vocab)

    def get_vocab_size(self) -> int:
        """Return the vocabulary size as a method call.

        Returns:
            Number of tokens in the vocabulary.
        """
        return len(self.vocab)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> BPETokenizer:
        """Load a tokenizer from serialized vocabulary and merge files.

        Special tokens absent from the vocabulary are appended with fresh ids,
        so loading the same artifacts with different special-token lists stays
        consistent.

        Args:
            vocab_filepath: Path to the JSON vocabulary file.
            merges_filepath: Path to the whitespace-separated merges file.
            special_tokens: Special tokens to register.

        Returns:
            The loaded tokenizer.
        """
        with open(vocab_filepath, encoding="utf-8") as f:
            raw_vocab: dict[str, str] = json.load(f)

        vocab: dict[int, bytes] = {
            int(idx): token_str.encode("latin-1") for idx, token_str in raw_vocab.items()
        }

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\r\n")
                if not line:
                    continue
                parts = line.split(" ")
                if len(parts) == 2:
                    merges.append((parts[0].encode("latin-1"), parts[1].encode("latin-1")))

        special_tokens = special_tokens or []
        existing_byte_values = set(vocab.values())

        for token in special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in existing_byte_values:
                vocab[len(vocab)] = token_bytes
                existing_byte_values.add(token_bytes)

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def save(
        self,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
    ) -> None:
        """Write the vocabulary as JSON and the merges as one pair per line.

        Args:
            vocab_filepath: Destination for the vocabulary.
            merges_filepath: Destination for the merge rules.
        """
        Path(vocab_filepath).parent.mkdir(parents=True, exist_ok=True)
        Path(merges_filepath).parent.mkdir(parents=True, exist_ok=True)

        serializable_vocab = {idx: b.decode("latin-1") for idx, b in self.vocab.items()}
        with open(vocab_filepath, "w", encoding="utf-8") as f:
            json.dump(serializable_vocab, f, ensure_ascii=False, indent=2)

        with open(merges_filepath, "w", encoding="utf-8") as f:
            for p0, p1 in self.merges:
                f.write(f"{p0.decode('latin-1')} {p1.decode('latin-1')}\n")

    @classmethod
    def from_vocab_and_merges(
        cls,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> BPETokenizer:
        """Build a tokenizer from in-memory artifacts, copying the vocabulary.

        Args:
            vocab: Mapping from token id to byte string.
            merges: Ordered merge rules.
            special_tokens: Special tokens to register.

        Returns:
            The new tokenizer.
        """
        vocab_copy = dict(vocab)
        special_tokens = special_tokens or []
        existing_byte_values = set(vocab_copy.values())

        for token in special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in existing_byte_values:
                vocab_copy[len(vocab_copy)] = token_bytes
                existing_byte_values.add(token_bytes)

        return cls(vocab=vocab_copy, merges=list(merges), special_tokens=special_tokens)

    @classmethod
    def train(
        cls,
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str] | None = None,
        save_vocab_path: str | os.PathLike | None = None,
        save_merges_path: str | os.PathLike | None = None,
        **kwargs,
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """Train BPE on a text file.

        Merges are chosen greedily by pair frequency, recomputing counts only
        for the chunks a merge touched. Special tokens are excluded from the
        counted text and appended to the vocabulary afterwards.

        Args:
            input_path: Text file to train on.
            vocab_size: Target vocabulary size, including the 256 base bytes
                and the special tokens.
            special_tokens: Tokens reserved as indivisible units.
            save_vocab_path: Where to write the vocabulary, if anywhere.
            save_merges_path: Where to write the merges, if anywhere. Both save
                paths must be given for anything to be written.
            **kwargs: Ignored; accepted so callers can pass extra options.

        Returns:
            The trained ``(vocab, merges)`` pair.

        Raises:
            FileNotFoundError: If ``input_path`` does not exist.
            ValueError: If ``vocab_size`` cannot fit the base and special tokens.
        """
        path = Path(input_path)
        if not path.is_file():
            raise FileNotFoundError(f"File Not Found: {path}")

        special_tokens = special_tokens or []
        num_merges = vocab_size - len(cls._BASE_TOKENS) - len(special_tokens)
        if num_merges < 0:
            raise ValueError(
                f"vocab_size ({vocab_size}) is too small to accommodate 256 base "
                f"bytes and {len(special_tokens)} special tokens"
            )

        vocab: dict[int, bytes] = cls._BASE_TOKENS.copy()
        merges: list[tuple[bytes, bytes]] = []

        special_set = set(special_tokens)
        if special_tokens:
            sorted_tokens = sorted(special_tokens, key=len, reverse=True)
            escaped = [regex.escape(t) for t in sorted_tokens]
            special_re = regex.compile(f"({'|'.join(escaped)})")
        else:
            special_re = None

        chunks = cls._pretokenize_file(path, special_re=special_re, special_set=special_set)

        chunk_counter = Counter(tuple(chunk) for chunk in chunks)
        pair_counter: Counter[tuple[bytes, bytes]] = Counter()
        pair_to_chunks: defaultdict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(
            set
        )

        for chunk, freq in chunk_counter.items():
            for i in range(len(chunk) - 1):
                pair = (chunk[i], chunk[i + 1])
                pair_counter[pair] += freq
                pair_to_chunks[pair].add(chunk)

        for _ in range(num_merges):
            if not pair_counter:
                break

            best_pair = max(pair_counter, key=lambda p: (pair_counter[p], p[0], p[1]))

            if pair_counter[best_pair] == 0:
                break

            new_token = best_pair[0] + best_pair[1]
            merges.append(best_pair)
            vocab[len(vocab)] = new_token

            affected_chunks = list(pair_to_chunks.pop(best_pair, set()))

            for chunk in affected_chunks:
                freq = chunk_counter.pop(chunk, 0)
                if freq == 0:
                    continue

                # Remove the old chunk's contribution to every pair count.
                for i in range(len(chunk) - 1):
                    p = (chunk[i], chunk[i + 1])
                    pair_counter[p] -= freq
                    if pair_counter[p] == 0:
                        pair_counter.pop(p, None)

                    if p in pair_to_chunks:
                        pair_to_chunks[p].discard(chunk)
                        if not pair_to_chunks[p]:
                            del pair_to_chunks[p]

                merged_chunk = tuple(cls.merge(list(chunk), best_pair))
                chunk_counter[merged_chunk] += freq

                # Add the merged chunk's contribution back.
                for i in range(len(merged_chunk) - 1):
                    p = (merged_chunk[i], merged_chunk[i + 1])
                    pair_counter[p] += freq
                    pair_to_chunks[p].add(merged_chunk)

        for token in special_tokens:
            vocab[len(vocab)] = token.encode("utf-8")

        if save_vocab_path is not None and save_merges_path is not None:
            tokenizer_instance = cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
            tokenizer_instance.save(save_vocab_path, save_merges_path)

        return vocab, merges

    @staticmethod
    def _pretokenize_text(
        text: str,
        special_re: regex.Pattern | None = None,
        special_set: set[str] | None = None,
    ) -> list[list[bytes]]:
        """Split text into pre-token chunks of UTF-8 bytes.

        Special tokens are kept as single-token chunks; everything else is
        split with the GPT-2 pre-tokenization pattern and then broken into
        individual bytes.

        Args:
            text: Text to segment.
            special_re: Compiled pattern matching any special token.
            special_set: The special tokens themselves.

        Returns:
            One list of byte chunks per pre-token.
        """
        if not text:
            return []

        chunks = special_re.split(text) if special_re else [text]
        pretokenized_chunks: list[list[bytes]] = []
        special_set = special_set or set()

        for chunk in chunks:
            if not chunk:
                continue

            if chunk in special_set:
                pretokenized_chunks.append([chunk.encode("utf-8")])
                continue

            for match in BPETokenizer._GPT2_SPLIT_RE.findall(chunk):
                byte_sequence = [bytes([b]) for b in match.encode("utf-8")]
                pretokenized_chunks.append(byte_sequence)

        return pretokenized_chunks

    @staticmethod
    def _pretokenize_file(
        input_path: str | os.PathLike,
        special_re: regex.Pattern | None = None,
        special_set: set[str] | None = None,
    ) -> list[list[bytes]]:
        """Read a whole text file and pre-tokenize its contents.

        Args:
            input_path: Text file to read.
            special_re: Compiled pattern matching any special token.
            special_set: The special tokens themselves.

        Returns:
            One list of byte chunks per pre-token.
        """
        text = Path(input_path).read_text(encoding="utf-8")
        return BPETokenizer._pretokenize_text(text, special_re=special_re, special_set=special_set)

    @staticmethod
    def merge(chunk: list[bytes], pair: tuple[bytes, bytes]) -> list[bytes]:
        """Merge every non-overlapping occurrence of ``pair`` in ``chunk``.

        Args:
            chunk: Sequence of byte tokens.
            pair: Adjacent token pair to combine.

        Returns:
            A new sequence in which each occurrence became one token.
        """
        left, right = pair
        target = left + right
        merged: list[bytes] = []
        index = 0
        chunk_len = len(chunk)

        while index < chunk_len:
            if index + 1 < chunk_len and chunk[index] == left and chunk[index + 1] == right:
                merged.append(target)
                index += 2
            else:
                merged.append(chunk[index])
                index += 1

        return merged

    def encode(self, text: str) -> list[int]:
        """Encode text into token ids.

        Args:
            text: Text to encode.

        Returns:
            The token id sequence.
        """
        chunks = self._pretokenize_text(
            text, special_re=self._special_re, special_set=self._special_set
        )
        for rule in self.merges:
            chunks = [self.merge(chunk, rule) for chunk in chunks]
        return [self.inverse_vocab[token] for chunk in chunks for token in chunk]

    def decode(self, ids: list[int]) -> str:
        """Decode token ids back into text.

        Invalid UTF-8 sequences are replaced rather than raising, so a
        truncated or corrupted id list still produces output.

        Args:
            ids: Token ids to decode.

        Returns:
            The decoded text.
        """
        tokens = [self.vocab[id] for id in ids]
        return b"".join(tokens).decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Encode an iterable of strings, yielding token ids one at a time.

        Streaming the ids keeps memory bounded when encoding a corpus that does
        not fit in RAM.

        Args:
            iterable: Strings to encode, e.g. lines of a file.

        Yields:
            One token id per iteration.
        """
        for text in iterable:
            if not text:
                continue

            chunks = self._pretokenize_text(
                text,
                special_re=self._special_re,
                special_set=self._special_set,
            )

            for rule in self.merges:
                chunks = [self.merge(chunk, rule) for chunk in chunks]

            for chunk in chunks:
                for token in chunk:
                    yield self.inverse_vocab[token]

    def encode_to_bin(
        self,
        input_text_path: str | os.PathLike,
        output_bin_path: str | os.PathLike,
        chunk_lines: int = 500,
        dtype: type = np.uint16,
        log_interval_lines: int = 1000,
    ) -> int:
        """Encode a text file into a flat binary token stream.

        Lines are batched before encoding to amortize per-call overhead, then
        written as a raw array of ``dtype``. The result is consumed by
        ``numpy.memmap`` at training time.

        Args:
            input_text_path: Text file to encode.
            output_bin_path: Destination binary file.
            chunk_lines: Lines to buffer before each encode call.
            dtype: Numpy dtype for the output array.
            log_interval_lines: How many processed lines between progress prints.

        Returns:
            Total number of tokens written.
        """
        total_tokens = 0
        total_processed_lines = 0
        last_log_lines = 0

        input_path = Path(input_text_path)
        output_path = Path(output_bin_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[Data Prep] Encoding {input_path.name} -> {output_path.name}")
        t0 = time.time()

        with open(input_path, encoding="utf-8") as in_f, open(output_path, "wb") as out_f:
            lines = []
            for line in in_f:
                lines.append(line)
                if len(lines) >= chunk_lines:
                    text_chunk = "".join(lines)
                    token_ids = self.encode(text_chunk)
                    if token_ids:
                        np.array(token_ids, dtype=dtype).tofile(out_f)
                        total_tokens += len(token_ids)

                    total_processed_lines += len(lines)
                    lines.clear()

                    if total_processed_lines - last_log_lines >= log_interval_lines:
                        elapsed = time.time() - t0
                        speed = total_tokens / max(elapsed, 1e-6)
                        print(
                            f"  -> Processed {total_processed_lines:8d} lines | "
                            f"Tokens: {total_tokens:10,d} | "
                            f"Speed: {speed:8,.0f} tok/s"
                        )
                        last_log_lines = total_processed_lines

            # Flush the final partial batch.
            if lines:
                text_chunk = "".join(lines)
                token_ids = self.encode(text_chunk)
                if token_ids:
                    np.array(token_ids, dtype=dtype).tofile(out_f)
                    total_tokens += len(token_ids)
                total_processed_lines += len(lines)

        total_time = time.time() - t0
        final_speed = total_tokens / max(total_time, 1e-6)
        print(
            f"[Data Prep] Complete! Total lines: {total_processed_lines:,} | "
            f"Tokens: {total_tokens:,} | Time: {total_time:.2f}s ({final_speed:,.0f} tok/s)\n"
        )

        return total_tokens
