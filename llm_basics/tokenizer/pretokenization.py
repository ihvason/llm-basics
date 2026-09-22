"""Split a large text file into chunks that can be pre-tokenized independently.

Useful for parallel corpus preprocessing: each chunk can be counted in its own
process and the per-chunk results merged afterwards.
"""

import os
from typing import BinaryIO


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """Find byte offsets that divide a file into roughly equal chunks.

    Each boundary is snapped forward to the next occurrence of
    ``split_special_token`` so that no chunk splits a document. Overlapping
    boundaries are collapsed, so fewer chunks than requested may be returned.

    Args:
        file: Open binary file handle; its position is modified.
        desired_num_chunks: Rough number of chunks to produce.
        split_special_token: Byte string marking a safe split point, such as
            ``b"<|endoftext|>"``.

    Returns:
        Sorted, de-duplicated byte offsets, starting at 0 and ending at the
        file size.

    Raises:
        AssertionError: If ``split_special_token`` is not a byte string.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Total file size in bytes.
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Uniformly spaced first guesses; the final boundary is the end of the file.
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead in 4 KiB steps while searching.

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)

            if mini_chunk == b"":
                # Reached EOF before finding the token: end this chunk at EOF.
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def example(text_path: str, num_processes: int = 4) -> None:
    """Show how to chunk a corpus for multi-process pre-tokenization.

    Each ``(start, end)`` pair can be dispatched to a worker that decodes its
    slice and counts pre-tokens.

    Args:
        text_path: Path to the raw corpus.
        num_processes: Number of chunks to split the corpus into.
    """
    with open(text_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Count pre-tokens in `chunk` here, or hand it to a worker process.
            _ = chunk


if __name__ == "__main__":
    raise SystemExit(
        "This module only exposes functions; see `example(text_path)` for usage. "
        "To train a tokenizer use llm_basics.tokenizer.tokenizer.BPETokenizer.train."
    )
