"""
BitDiffusion a4.8 — English-only data preparation (~40B tokens)

OUTPUT FORMAT  (matches StreamingJsonlDataset in data.py — do NOT change)
  Each line: {"text": "..."}

RECOMMENDED TRAINING CONFIGS
──────────────────────────────────────────────────────────────────
1B MODEL  (~30B tokens — 30 tokens/param, uses full data mix)
  hidden_dim=2048  n_layers=16  n_heads=16  head_dim=128
  ffn_dim=8192     seq_len=2048  batch_size=16  grad_accum=8
  peak_lr=1.5e-4   warmup_steps=2000  max_steps=115_000
  a4_warmup_fraction=0.10  gradient_checkpointing=True
  → 115K × (16 × 2048 × 8) = ~30.1B tokens

  Budget variant (~13B tokens / $200 / 7 days A100 40GB spot):
    max_steps=50_000 → ~13.1B tokens (undertrained but functional)

2B MODEL  (~43B tokens — 21 tokens/param, full data mix)
  hidden_dim=2048  n_layers=32  n_heads=16  head_dim=128
  ffn_dim=8192     seq_len=2048  batch_size=8   grad_accum=16
  peak_lr=1e-4     warmup_steps=2000  max_steps=165_000
  a4_warmup_fraction=0.10  gradient_checkpointing=True
  → 165K × (8 × 2048 × 16) = ~43.3B tokens

  Budget variant (~20B tokens / $450):
    max_steps=76_000 → ~19.9B tokens
──────────────────────────────────────────────────────────────────

DATASET MIX  (~40B tokens, English only)
┌─────────────────┬──────────────────────────────────────────────────────┬───────┐
│ Slug            │ Source                                               │Tokens │
├─────────────────┼──────────────────────────────────────────────────────┼───────┤
│ fineweb_edu     │ HuggingFaceFW/fineweb-edu (sample-100BT)             │  15B  │
│                 │ Llama-3-70B scored educational web — gold standard   │       │
├─────────────────┼──────────────────────────────────────────────────────┼───────┤
│ dclm            │ HuggingFaceFW/dclm_100BT                             │   8B  │
│                 │ Model-filtered CC; beats SlimPajama/RefinedWeb/Dolma │       │
├─────────────────┼──────────────────────────────────────────────────────┼───────┤
│ open_web_math   │ open-web-math/open-web-math (14.7B total)            │   7B  │
│                 │ Best-in-class math/STEM web corpus                   │       │
├─────────────────┼──────────────────────────────────────────────────────┼───────┤
│ cosmopedia      │ HuggingFaceTB/cosmopedia (~30B total)                │   4B  │
│                 │ Synthetic textbook/story — strong reasoning signal   │       │
├─────────────────┼──────────────────────────────────────────────────────┼───────┤
│ wikipedia_en    │ wikimedia/wikipedia 20231101.en (~4.4B total)        │   2B  │
│                 │ Encyclopedic ground-truth factual knowledge          │       │
├─────────────────┼──────────────────────────────────────────────────────┼───────┤
│ finepdfs        │ HuggingFaceFW/finepdfs_100BT                        │   2B  │
│                 │ PDF-extracted academic papers, books, tech docs      │       │
├─────────────────┼──────────────────────────────────────────────────────┼───────┤
│ mathcode_pile   │ MathGenie/MathCode-Pile (19.2B total)               │   2B  │
│                 │ Math web + textbooks + model-synth + math code       │       │
└─────────────────┴──────────────────────────────────────────────────────┴───────┘
                                                               TOTAL:    40B
"""

import json
import os
import random
import tempfile
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

TOKENIZER_NAME = "Qwen/Qwen-tokenizer"
VAL_RATIO      = 0.005   # 0.5% val — plenty of data, keep val small
SEED           = 42
MIN_TOKENS     = 64
MAX_TOKENS     = 8192
CHUNK_OVERLAP  = 128     # overlap between chunks when splitting long docs
BATCH_SIZE     = 128     # batch tokenisation — much faster than one-at-a-time
SHUFFLE_BUCKET = 500_000 # lines held in memory during streaming shuffle

TRAIN_DIR   = Path("data/train")
VAL_DIR     = Path("data/val")
SHARD_DIR   = Path("data/hf_shards")
STATE_PATH  = SHARD_DIR / "progress.json"
FINAL_TRAIN = TRAIN_DIR / "hf_mix_train.jsonl"
FINAL_VAL   = VAL_DIR   / "hf_mix_val.jsonl"

# ──────────────────────────────────────────────────────────────────────────────
# DATASETS
# All use text_field="text" (confirmed for fineweb_edu, dclm, open_web_math,
# cosmopedia, wikipedia_en, finepdfs).
# mathcode_pile field assumed "text" — if it produces 0 docs, change to "content"
# ──────────────────────────────────────────────────────────────────────────────
DATASETS = [
    {
        "slug":          "fineweb_edu",
        "path":          "HuggingFaceFW/fineweb-edu",
        "name":          "sample-100BT",
        "split":         "train",
        "target_tokens": 15_000_000_000,
        "text_field":    "text",
    },
    {
        "slug":          "dclm",
        "path":          "HuggingFaceFW/dclm_100BT",
        "split":         "train",
        "target_tokens": 8_000_000_000,
        "text_field":    "text",
    },
    {
        "slug":          "open_web_math",
        "path":          "open-web-math/open-web-math",
        "split":         "train",
        "target_tokens": 7_000_000_000,
        "text_field":    "text",
    },
    {
        "slug":          "cosmopedia",
        "path":          "HuggingFaceTB/cosmopedia",
        "split":         "train",
        "target_tokens": 4_000_000_000,
        "text_field":    "text",
    },
    {
        "slug":          "wikipedia_en",
        "path":          "wikimedia/wikipedia",
        "name":          "20231101.en",
        "split":         "train",
        "target_tokens": 2_000_000_000,
        "text_field":    "text",
    },
    {
        "slug":          "finepdfs",
        "path":          "HuggingFaceFW/finepdfs_100BT",
        "split":         "train",
        "target_tokens": 2_000_000_000,
        "text_field":    "text",
    },
    {
        # 19.2B tokens: math web pages + textbooks + model-synthesised + math code
        # text_field assumed "text" — if docs=0 after start, change to "content"
        "slug":          "mathcode_pile",
        "path":          "MathGenie/MathCode-Pile",
        "split":         "train",
        "target_tokens": 2_000_000_000,
        "text_field":    "text",
    },

    # ── CODE ──────────────────────────────────────────────────────────────────
    # StarCoderData uses "content" field (not "text") and data_dir for language.
    # Python: ~35B tokens available.  JS/TS: ~24B tokens available.
    # Code improves reasoning, instruction-following, and structured generation.
    {
        "slug":          "starcoder_python",
        "path":          "bigcode/starcoderdata",
        "data_dir":      "python",
        "split":         "train",
        "target_tokens": 2_000_000_000,
        "text_field":    "content",
    },
    {
        "slug":          "starcoder_js",
        "path":          "bigcode/starcoderdata",
        "data_dir":      "javascript",
        "split":         "train",
        "target_tokens": 1_000_000_000,
        "text_field":    "content",
    },
]
# Total target: 15B + 8B + 7B + 4B + 2B + 2B + 2B + 2B + 1B = 43B tokens


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def ensure_dirs():
    for d in (TRAIN_DIR, VAL_DIR, SHARD_DIR):
        d.mkdir(parents=True, exist_ok=True)


def shard_paths(spec):
    return (
        SHARD_DIR / f"{spec['slug']}.train.jsonl",
        SHARD_DIR / f"{spec['slug']}.val.jsonl",
    )


def load_state():
    if not STATE_PATH.exists():
        return {}
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def save_state(state):
    STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_stream(spec):
    kwargs = {"path": spec["path"], "split": spec["split"], "streaming": True}
    if spec.get("name"):     kwargs["name"]     = spec["name"]
    if spec.get("data_dir"): kwargs["data_dir"] = spec["data_dir"]
    return load_dataset(**kwargs)


def iter_texts(spec):
    field = spec.get("text_field", "text")
    fallback = "content" if field == "text" else "text"
    detected = False
    for row in load_stream(spec):
        text = row.get(field)
        if not detected and text is None:
            text = row.get(fallback)
            if text is not None:
                print(f"  auto-detected text_field='{fallback}' for {spec['slug']}")
                field = fallback
        detected = True
        if isinstance(text, str):
            text = text.strip()
        if text:
            yield text


def needs_rebuild(spec, state):
    slug = spec["slug"]
    src  = state.get(slug, {})
    t, v = shard_paths(spec)
    if not src.get("complete"):           return True
    if not t.exists() or not v.exists(): return True
    if src.get("target_tokens", 0) != spec["target_tokens"]:
        print(f"  {slug}: target changed → rebuilding")
        return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
# BUILD ONE SOURCE SHARD
# Uses batched tokenisation (BATCH_SIZE docs at a time) for speed.
# ──────────────────────────────────────────────────────────────────────────────
def build_source(tokenizer, spec, state):
    slug = spec["slug"]
    train_shard, val_shard = shard_paths(spec)

    if not needs_rebuild(spec, state):
        s = state[slug]
        print(f"  skip {slug:<20} {s['tokens']/1e9:.2f}B tokens already collected")
        return s

    rng    = random.Random(f"{SEED}:{slug}")
    target = spec["target_tokens"]
    docs = written = train_tokens = val_tokens = 0

    print(f"\n  building {slug}  (target {target/1e9:.0f}B tokens)")

    buf = []

    with train_shard.open("w", encoding="utf-8") as tf, \
         val_shard.open("w",   encoding="utf-8") as vf:

        def flush(buf):
            nonlocal docs, written, train_tokens, val_tokens
            if not buf:
                return
            enc = tokenizer(
                buf,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                truncation=False,
                verbose=False,
            )
            for text, ids in zip(buf, enc["input_ids"]):
                tc = len(ids)
                if tc < MIN_TOKENS:
                    continue
                # chunk long docs with overlap instead of dropping them
                if tc > MAX_TOKENS:
                    step = MAX_TOKENS - CHUNK_OVERLAP
                    for start in range(0, tc, step):
                        chunk_ids = ids[start : start + MAX_TOKENS]
                        if len(chunk_ids) < MIN_TOKENS:
                            break
                        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
                        ctc = len(chunk_ids)
                        line = json.dumps({"text": chunk_text}, ensure_ascii=False) + "\n"
                        if rng.random() < VAL_RATIO:
                            vf.write(line);  val_tokens   += ctc
                        else:
                            tf.write(line);  train_tokens += ctc
                        written += ctc
                        docs    += 1
                else:
                    line = json.dumps({"text": text}, ensure_ascii=False) + "\n"
                    if rng.random() < VAL_RATIO:
                        vf.write(line);  val_tokens   += tc
                    else:
                        tf.write(line);  train_tokens += tc
                    written += tc
                    docs    += 1

        for text in iter_texts(spec):
            buf.append(text)
            if len(buf) >= BATCH_SIZE:
                flush(buf); buf = []
                if docs % 10_000 == 0 and docs > 0:
                    print(f"    {slug}: {docs:>8,} docs  {written/1e9:.3f}B / {target/1e9:.0f}B tokens")
                if written >= target:
                    break
        flush(buf)

    # warn if almost nothing came through (wrong field name)
    if docs < 100:
        print(f"  WARNING: {slug} produced only {docs} docs — check text_field value")

    result = {
        "complete":      True,
        "target_tokens": target,
        "docs":          docs,
        "tokens":        written,
        "train_tokens":  train_tokens,
        "val_tokens":    val_tokens,
    }
    state[slug] = result
    save_state(state)
    print(f"  done {slug:<20} {written/1e9:.3f}B tokens  {docs:,} docs")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL SHUFFLE  (prevents any domain clustering in training)
# ──────────────────────────────────────────────────────────────────────────────
def assemble_shuffled(state):
    """Streaming bucket-shuffle: never holds more than SHUFFLE_BUCKET lines in RAM."""
    rng = random.Random(SEED)

    for split, final in [("train", FINAL_TRAIN), ("val", FINAL_VAL)]:
        print(f"\nassembling {split}...")

        # count lines and collect shard paths
        shards = []
        total_lines = 0
        for spec in DATASETS:
            shard = shard_paths(spec)[0 if split == "train" else 1]
            if not shard.exists():
                print(f"  WARNING: missing shard {shard.name} — skipping")
                continue
            n = sum(1 for line in shard.open(encoding="utf-8") if line.strip())
            print(f"  {spec['slug']:<20} {n:>8,} docs")
            shards.append(shard)
            total_lines += n

        if total_lines == 0:
            print(f"  WARNING: no data for {split}")
            continue

        # assign each line to a random bucket (temp file)
        n_buckets = max(1, total_lines // SHUFFLE_BUCKET)
        tmp_dir = tempfile.mkdtemp(prefix="hf_shuffle_")
        bucket_files = [
            open(os.path.join(tmp_dir, f"bucket_{i}.jsonl"), "w", encoding="utf-8")
            for i in range(n_buckets)
        ]

        print(f"  distributing {total_lines:,} docs into {n_buckets} buckets...")
        for shard in shards:
            with shard.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    bucket_files[rng.randrange(n_buckets)].write(line)

        for bf in bucket_files:
            bf.close()

        # shuffle each bucket in memory and write to final output
        print(f"  writing {final}...")
        written = 0
        with final.open("w", encoding="utf-8") as out:
            for i in range(n_buckets):
                bp = os.path.join(tmp_dir, f"bucket_{i}.jsonl")
                with open(bp, encoding="utf-8") as bf:
                    bucket = bf.readlines()
                rng.shuffle(bucket)
                out.writelines(bucket)
                written += len(bucket)
                os.remove(bp)

        os.rmdir(tmp_dir)
        gb = final.stat().st_size / 1e9
        print(f"  wrote {final}  ({gb:.1f} GB, {written:,} docs)")


def print_summary(state):
    total = sum(v.get("tokens", 0) for v in state.values())
    print(f"\n{'─'*58}")
    print(f"  {'Dataset':<20} {'Tokens':>10}  {'Docs':>10}  St")
    print(f"{'─'*58}")
    for spec in DATASETS:
        s    = state.get(spec["slug"], {})
        flag = "✓" if s.get("complete") else "·"
        print(f"  {flag} {spec['slug']:<19} {s.get('tokens',0)/1e9:>8.2f}B  {s.get('docs',0):>10,}")
    print(f"{'─'*58}")
    print(f"  {'TOTAL':<20} {total/1e9:>8.2f}B")
    print(f"{'─'*58}\n")


# ──────────────────────────────────────────────────────────────────────────────
def main():
    ensure_dirs()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    state     = load_state()

    for spec in DATASETS:
        build_source(tokenizer, spec, state)

    print_summary(state)
    assemble_shuffled(state)


if __name__ == "__main__":
    main()
