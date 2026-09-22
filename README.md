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

## Development

```bash
pytest                              # test suite
ruff check .                        # lint
ty check src                        # type check
```

## Roadmap

Hyperparameter experiments, tracking validation loss, perplexity and tokens per
second.

- **Learning rate** — peak LR sweep, warmup length, decay schedule
- **Architecture** — depth/width scaling, head count, feed-forward ratio, RoPE theta
- **Optimization** — batch size against learning rate, betas, weight decay, autocast dtype
- **Tokenizer** — vocabulary size, special tokens, cross-domain comparison
- **Data** — context length and dataset size against step count
- **Infrastructure** — KV cache, multi-GPU training
