import json
import random
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

TOKENIZER_NAME = "Qwen/Qwen-tokenizer"
VAL_RATIO = 0.01
SEED = 42
MIN_TOKENS = 32
MAX_TOKENS = 32768

TRAIN_DIR = Path("data/train")
VAL_DIR = Path("data/val")
SHARD_DIR = Path("data/hf_shards")
STATE_PATH = SHARD_DIR / "progress.json"
FINAL_TRAIN = TRAIN_DIR / "hf_mix_train.jsonl"
FINAL_VAL = VAL_DIR / "hf_mix_val.jsonl"

# Healthy English + Korean pretraining mix focused on education, math, and reference text.
# This stays in plain {"text": "..."} format for the current training pipeline.
DATASETS = [
    {
        "slug": "fineweb_edu_en",
        "path": "HuggingFaceFW/fineweb_edu_100BT",
        "split": "train",
        "target_tokens": 180_000_000,
    },
    {
        "slug": "open_web_math_en",
        "path": "open-web-math/open-web-math",
        "split": "train",
        "target_tokens": 90_000_000,
    },
    {
        "slug": "finepdfs_edu_en",
        "path": "HuggingFaceFW/finepdfs_edu_100BT",
        "split": "train",
        "target_tokens": 40_000_000,
    },
    {
        "slug": "wikipedia_en",
        "path": "wikimedia/wikipedia",
        "name": "20231101.en",
        "split": "train",
        "target_tokens": 50_000_000,
    },
    {
        "slug": "korean_webtext_edu",
        "path": "eliceai/korean-webtext-edu",
        "split": "train",
        "target_tokens": 90_000_000,
    },
    {
        "slug": "fineweb2_edu_ko",
        "path": "minpeter/fineweb-2-edu-korean",
        "split": "train",
        "target_tokens": 90_000_000,
    },
    {
        "slug": "wikipedia_ko",
        "path": "wikimedia/wikipedia",
        "name": "20231101.ko",
        "split": "train",
        "target_tokens": 30_000_000,
    },
    {
        "slug": "korean_webtext",
        "path": "HAERAE-HUB/KOREAN-WEBTEXT",
        "split": "train",
        "target_tokens": 10_000_000,
    },
]


def ensure_dirs():
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)


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
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def load_stream(spec):
    kwargs = {
        "path": spec["path"],
        "split": spec["split"],
        "streaming": True,
    }
    if spec.get("name"):
        kwargs["name"] = spec["name"]
    return load_dataset(**kwargs)


def iter_texts(spec):
    for row in load_stream(spec):
        text = row.get("text")
        if isinstance(text, str):
            text = text.strip()
        if text:
            yield text


def count_tokens(tokenizer, text):
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        verbose=False,
    )
    return len(encoded["input_ids"])


def build_source(tokenizer, spec, state):
    slug = spec["slug"]
    label = f"{spec['path']}" + (f" / {spec['name']}" if spec.get("name") else "")
    train_shard, val_shard = shard_paths(spec)
    source_state = state.get(slug, {})

    if source_state.get("complete") and train_shard.exists() and val_shard.exists():
        print(f"Skipping {label}; shard already complete.")
        return source_state

    rng = random.Random(f"{SEED}:{slug}")
    target = spec["target_tokens"]
    docs = 0
    written = 0
    train_tokens = 0
    val_tokens = 0
    skipped_short = 0
    skipped_long = 0

    print(f"Starting {label} with target {target:,} tokens")

    with train_shard.open("w", encoding="utf-8") as train_f, val_shard.open("w", encoding="utf-8") as val_f:
        for text in iter_texts(spec):
            token_count = count_tokens(tokenizer, text)
            if token_count < MIN_TOKENS:
                skipped_short += 1
                continue
            if token_count > MAX_TOKENS:
                skipped_long += 1
                continue

            line = json.dumps({"text": text}, ensure_ascii=False) + "\n"
            if rng.random() < VAL_RATIO:
                val_f.write(line)
                val_tokens += token_count
            else:
                train_f.write(line)
                train_tokens += token_count

            written += token_count
            docs += 1

            if docs % 1000 == 0:
                print(f"  {slug}: docs={docs:,}, tokens={written:,}")
                state[slug] = {
                    "complete": False,
                    "docs": docs,
                    "tokens": written,
                    "train_tokens": train_tokens,
                    "val_tokens": val_tokens,
                    "skipped_short": skipped_short,
                    "skipped_long": skipped_long,
                }
                save_state(state)

            if written >= target:
                break

    result = {
        "complete": True,
        "docs": docs,
        "tokens": written,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "skipped_short": skipped_short,
        "skipped_long": skipped_long,
    }
    state[slug] = result
    save_state(state)

    print(
        f"Finished {label}: docs={docs:,}, tokens={written:,}, "
        f"skipped_short={skipped_short:,}, skipped_long={skipped_long:,}"
    )
    return result


def concat_file(src, dst_f):
    if not src.exists():
        return
    with src.open("r", encoding="utf-8") as src_f:
        for line in src_f:
            dst_f.write(line)


def assemble_final_files(state):
    FINAL_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    FINAL_VAL.parent.mkdir(parents=True, exist_ok=True)

    with FINAL_TRAIN.open("w", encoding="utf-8") as train_out, FINAL_VAL.open("w", encoding="utf-8") as val_out:
        for spec in DATASETS:
            source_state = state.get(spec["slug"], {})
            if not source_state.get("complete"):
                print(f"Skipping merge for {spec['slug']}; source is incomplete.")
                continue
            train_shard, val_shard = shard_paths(spec)
            concat_file(train_shard, train_out)
            concat_file(val_shard, val_out)


def print_summary(state):
    train_total = 0
    val_total = 0
    complete = 0
    for spec in DATASETS:
        source_state = state.get(spec["slug"], {})
        if source_state.get("complete"):
            complete += 1
            train_total += source_state.get("train_tokens", 0)
            val_total += source_state.get("val_tokens", 0)

    print(f"Completed sources: {complete}/{len(DATASETS)}")
    print(f"Train tokens written: {train_total:,}")
    print(f"Val tokens written:   {val_total:,}")
    print(f"Final train file: {FINAL_TRAIN}")
    print(f"Final val file:   {FINAL_VAL}")


def main():
    ensure_dirs()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    state = load_state()

    print(f"Tokenizer: {TOKENIZER_NAME}")
    print("Target token budget: 580,000,000")
    print(f"Shard directory: {SHARD_DIR}")

    for spec in DATASETS:
        build_source(tokenizer, spec, state)

    assemble_final_files(state)
    print_summary(state)


if __name__ == "__main__":
    main()
