# llm_basics

A decoder-only Transformer language model, written from scratch in PyTorch.

Attention, RMSNorm, SwiGLU, RoPE, the BPE tokenizer, the AdamW optimizer and
the training loop are all implemented directly. Nothing is delegated to
`torch.nn.Transformer` or another prebuilt model.

## Implemented

| Area | Contents |
| --- | --- |
| Tokenizer | Byte-level BPE: `train`, `encode`, `decode`, `encode_iterable`, `encode_to_bin`; GPT-2 style regex pre-tokenization; special tokens matched as whole units |
| Primitives | `Linear`, `Embedding`, `RMSNorm`, `SwiGLU` |
| Activations | Stable `softmax`, SiLU |
| Positions | `RotaryPositionalEmbedding` (RoPE) with precomputed cos/sin tables |
| Attention | `scaled_dot_product_attention`, `CausalMultiHeadSelfAttention` |
| Model | `TransformerBlock` (pre-norm with residuals), `TransformerLM` |
| Optimizer | Hand-written `AdamW`, cosine schedule with warmup, gradient clipping |
| Loss | Cross-entropy via log-sum-exp |
| Data | `get_batch` over a memory-mapped token array |
| Inference | Temperature, top-k and top-p sampling; autoregressive generation |
| Utilities | Config loading with deep merge, path resolution, seeding, device selection |

Not implemented: no KV cache, no weight tying, no distributed training, no beam
search or quantization.

## Layout

```
llm_basics/
├── modeling/     operators and TransformerLM
├── tokenizer/    BPE tokenizer and pre-tokenization
├── training/     loss, AdamW, schedule, checkpoints, logging
├── inference/    sampling and generation
└── utils/        paths, seed, device, dataloader, config
tests/            snapshot and reference tests
scripts/          command-line entry points
configs/          default.yaml
```

## Setup

Requires Python `>=3.12,<3.14`.

```bash
uv sync
```

## Usage

Paths have no defaults, so each step needs them explicitly.

```bash
# 1. Train a tokenizer and encode the corpus into token binaries
python scripts/prepare_data.py \
  --raw_train_path        '${DATA_ROOT}/corpus/train.txt' \
  --train_text_path       '${DATA_ROOT}/corpus/train.txt' \
  --train_bin_output_path '${DATA_ROOT}/tokenized/train.bin' \
  --vocab_output_path     '${DATA_ROOT}/tokenizer/vocab.json' \
  --merges_output_path    '${DATA_ROOT}/tokenizer/merges.txt'

# 2. Train the model
python scripts/run_train.py \
  --vocab_path     '${DATA_ROOT}/tokenizer/vocab.json' \
  --merges_path    '${DATA_ROOT}/tokenizer/merges.txt' \
  --train_bin_path '${DATA_ROOT}/tokenized/train.bin' \
  --checkpoint_dir '${CHECKPOINT_ROOT}' \
  --runs_dir       '${RUNS_ROOT}' \
  --max_iters 2000

# 3. Generate text
python scripts/run_generate.py \
  --checkpoint_path '${CHECKPOINT_ROOT}/latest.pt' \
  --vocab_path      '${DATA_ROOT}/tokenizer/vocab.json' \
  --merges_path     '${DATA_ROOT}/tokenizer/merges.txt' \
  --prompt "Once upon a time"
```

Note the single quotes around `${DATA_ROOT}`, `${CHECKPOINT_ROOT}` and
`${RUNS_ROOT}` on the command line: the shell would otherwise expand them (to
nothing, since they are not shell variables) before Python sees them. Quoting
passes the literal `${...}` through, and the config loader resolves it. In a
YAML file or a `.env`-style export the quoting is unnecessary.

The three roots are independent siblings. Redirecting one never moves another:

| Variable | Environment override | Default |
| --- | --- | --- |
| `${DATA_ROOT}` | `LLM_BASICS_DATA_ROOT` | `<repo>/data` |
| `${CHECKPOINT_ROOT}` | `LLM_BASICS_CHECKPOINT_ROOT` | `<repo>/checkpoints` |
| `${RUNS_ROOT}` | `LLM_BASICS_RUNS_ROOT` | `<repo>/runs` |

## Configuration

Three layers, later overriding earlier:

1. `configs/default.yaml` — loaded automatically
2. `--config <path>` — an experiment file with only the fields you change
3. Command-line flags

Write an experiment file anywhere you like and reuse it across the three steps:

```yaml
tokenizer:
  raw_train_path: ${DATA_ROOT}/corpus/train.txt
  vocab_output_path: ${DATA_ROOT}/tokenizer/vocab.json
  merges_output_path: ${DATA_ROOT}/tokenizer/merges.txt

data:
  # Text in: prepare_data.py reads these.
  train_text_path: ${DATA_ROOT}/corpus/train.txt
  # Token binaries out: prepare_data.py writes these.
  train_bin_output_path: ${DATA_ROOT}/tokenized/train.bin

model:
  d_model: 512
  num_layers: 8

train:
  # Token binaries in: run_train.py memory-maps these with dtype=uint16.
  # Normally the same file as data.train_bin_output_path above.
  train_bin_path: ${DATA_ROOT}/tokenized/train.bin
  checkpoint_dir: ${CHECKPOINT_ROOT}
  runs_dir: ${RUNS_ROOT}
  lr: 1.0e-4
```

```bash
python scripts/run_train.py --config my_experiment.yaml
python scripts/run_train.py --config my_experiment.yaml --max_iters 100
```

Two things to know:

- `device` exists in both the `train` and `generate` sections and has no flat
  key. Read `config.train.device` or `config.generate.device`.
- A missing path or a misspelled field raises an error naming the problem.

Data may live outside the repository:

```bash
export LLM_BASICS_DATA_ROOT=/mnt/volume/llm_data
```

Paths in a config may then use `${DATA_ROOT}/...`.

## Run artifacts

Training writes two trees, both namespaced by the experiment name:

```
${CHECKPOINT_ROOT}/
├── latest.pt                  # relative symlink to the newest checkpoint
└── <exp_name>/
    └── ckpt_iter_<N>.pt       # model + optimizer + iteration

${RUNS_ROOT}/
└── <exp_name>/
    ├── <exp_name>_config.json
    └── <exp_name>_metrics.jsonl
```

`<exp_name>` is `train.exp_name` when set, otherwise
`lm_L<layers>_D<d_model>_lr<lr>_<YYYYMMDDHHMM>` — for example
`lm_L2_D128_lr3e-4_202509221430`. An automatic name that is already taken gets a
`-2`, `-3`, … suffix, so two runs of the same configuration never share a
directory however close together they start. An explicitly supplied name is used
verbatim, which is what you want when resuming.

A checkpoint is saved every `save_interval` iterations and always on the final
iteration, so the last file is not necessarily a multiple of `save_interval`.
`latest.pt` is a *relative* symlink so the tree stays portable when the root
moves, and generation can target the newest run without naming it.

## Running on a server

```bash
git clone git@github.com:ihvason/llm-basics.git
cd llm-basics

# Optional. Defaults are <repo>/data, <repo>/checkpoints and
# <repo>/runs; point them at real disk on a server.
export LLM_BASICS_DATA_ROOT=/mnt/data/llm
export LLM_BASICS_CHECKPOINT_ROOT=/mnt/big/checkpoints
export LLM_BASICS_RUNS_ROOT=/mnt/logs/runs

# Writes ${DATA_ROOT}/tokenized/*.bin. Required: those are gitignored.
python scripts/prepare_data.py

python scripts/run_train.py

python scripts/run_generate.py
```
## Data

Download the TinyStories data and a subsample of OpenWebText:

```bash
cd data/

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz
```

Run these in `data/` (or wherever `LLM_BASICS_DATA_ROOT` points) and pass the
resulting paths to `scripts/prepare_data.py`.

## Tests

```bash
pytest                          # whole suite, ~52 s
pytest tests/test_model.py -q   # a single module
pytest -rs                      # show why tests were skipped
```

**48 tests: 46 pass, 2 skip.** See [Results](#results) for the last run.

| Module | Tests | Covers |
| --- | --- | --- |
| `test_tokenizer.py` | 25 | BPE encode/decode round-trips, special tokens, tiktoken parity, memory bounds |
| `test_model.py` | 13 | Every operator plus the full `TransformerLM` forward pass |
| `test_nn_utils.py` | 3 | `softmax`, cross-entropy, gradient clipping |
| `test_train_bpe.py` | 3 | BPE training vs. reference merges/vocab, and training speed |
| `test_optimizer.py` | 2 | AdamW and the cosine schedule with warmup |
| `test_data.py` | 1 | `get_batch` shapes, next-token shift and start-index sampling |
| `test_serialization.py` | 1 | Checkpoint save/load round-trip |

### How the suite is wired

`tests/adapters.py` is the seam between the assignment's canonical function names
(`run_linear`, `run_swiglu`, `run_multihead_self_attention`, `get_adamw_cls`, …)
and this project's API. Every test calls through it, so the stock CS336 tests run
unmodified against this implementation while the library keeps its own naming.

`tests/conftest.py` supplies the fixtures. Two matter most:

- `ts_state_dict` — a reference GPT-2 state dict plus its config, so each operator
  is fed the *real* weights of a known-good model instead of random ones. This is
  what makes a wrong implementation produce a visibly wrong number rather than
  merely a differently-initialised one.
- `numpy_snapshot` — loads `tests/_snapshots/*.npz` and asserts `rtol`/`atol`
  closeness, reporting exactly which keys are missing or unexpected.

`tests/fixtures/` holds reference corpora (English, German, addresses, TinyStories),
GPT-2 tokenizer artifacts and reference BPE output. `tests/_snapshots/` holds the
expected tensor outputs.

### What each module checks

**`test_tokenizer.py` — 25 tests.** Mostly parity: every round-trip test is paired
with a `*_matches_tiktoken` test asserting the same ids the reference tokenizer
produces. Coverage runs from the degenerate cases (empty string, one ASCII
character, one non-ASCII character) through mixed-Unicode strings, to real
fixtures (`address.txt`, `german.txt`, a TinyStories sample). Special-token
handling gets its own cases: a token followed by trailing newlines, a token
sandwiched between non-whitespace, and greedy matching of overlapping special
tokens. Four tests cover `encode_iterable` (streaming mode) round-trip and parity.
Two more bound memory usage under a 1 MB `RLIMIT_AS` — they skip off Linux.

**`test_model.py` — 13 tests.** One test per operator — `Linear`, `Embedding`,
`RMSNorm`, `SwiGLU`, RoPE, scaled dot-product attention (3D and 4D inputs),
multi-head attention with and without RoPE, and a single `TransformerBlock` — each
fed the reference weights and checked against a stored snapshot. `test_transformer_lm`
then runs the whole model end-to-end and compares logits, and
`test_transformer_lm_truncated_input` confirms a shorter-than-context-length input
still produces the expected result (catching a hard-coded context length).

**`test_nn_utils.py` — 3 tests.** `softmax` and the cross-entropy loss are each
checked against their PyTorch equivalent. `test_gradient_clipping` compares the
custom clip against `torch.nn.utils.clip_grad_norm_` on identical gradients, and
freezes one parameter to confirm it is excluded from the norm. (`test_silu_matches_pytorch`
lives in `test_model.py`, not here.)

**`test_train_bpe.py` — 3 tests.** `test_train_bpe` trains on a fixture corpus and
compares the resulting merges and vocabulary against the reference artifacts —
merges as an exact ordered list, vocab by key/value set. `test_train_bpe_special_tokens`
checks special tokens land in the vocabulary and are never split by merges, then
snapshots the result. `test_train_bpe_speed` requires training to finish in under
1.5 s, a budget the reference implementation clears in ~0.4 s; a naive
implementation takes several seconds and trips it.

**`test_optimizer.py` — 2 tests.** `test_adamw` runs 1000 steps and passes if the
weights match PyTorch's `AdamW` *or* the stored reference snapshot — the two differ
slightly because weight decay can be applied in equivalent-but-not-bit-identical
ways. `test_get_lr_cosine_schedule` pins 25 exact values across the warmup ramp,
the cosine decay and the post-cycle floor.

**`test_data.py` — 1 test.** `get_batch` is sampled 1000 times and must return
correctly shaped `(x, y)` with `y` shifted by exactly one, cover every valid start
index, and keep each index's frequency within ±5σ of uniform. That catches both an
off-by-one at the end of the array and a sampler that quietly ignores part of it.
It then asserts that an invalid device ordinal (`cuda:99`) raises.

**`test_serialization.py` — 1 test.** Trains a small net for 10 steps, saves a
checkpoint, loads it into a fresh model and optimizer, and asserts the iteration,
model weights and full optimizer state (moment estimates included) match.

### Results

```
$ pytest tests -q
46 passed, 2 skipped in 51.77s
```

Environment: Python 3.13.13, pytest 9.0.2, macOS / Apple Silicon. The suite runs on
CPU — only `run_train.py` selects the MPS device. The slowest tests are the four
`test_tokenizer.py` cases that encode the 5 MB TinyStories fixture, roughly 9 s each.

The only skips are the two memory-bound tests, both gated on
`sys.platform.startswith("linux")` because RLIMIT support is unreliable elsewhere:

| Test | Status | Why |
| --- | --- | --- |
| `test_encode_iterable_memory_usage` | skipped | Asserts streaming `encode_iterable` stays under 1 MB |
| `test_encode_memory_usage` | skipped | Also marked `xfail`: non-streaming `encode` is *expected* to exceed 1 MB |

To run them on Linux (or in a container), no change is needed — the platform guard
lets them through automatically.

## Roadmap

Hyperparameter experiments, tracking validation loss, perplexity and tokens per
second.

- **Learning rate** — peak LR sweep, warmup length, decay schedule
- **Architecture** — depth/width scaling, head count, feed-forward ratio, RoPE theta
- **Optimization** — batch size against learning rate, betas, weight decay, autocast dtype
- **Tokenizer** — vocabulary size, special tokens, cross-domain comparison
- **Data** — context length and dataset size against step count
- **Infrastructure** — KV cache, multi-GPU training
