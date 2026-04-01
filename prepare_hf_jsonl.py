"""
Prepare shuffled bilingual (EN+KO) JSONL training data for BitDiffusion a4.8.

Key improvements over v1:
  - Document-level shuffle across ALL sources before writing final files
  - Better English/Korean balance (55%/45%)
  - Added cosmopedia for structured educational/textbook content
  - Increased Korean reference text (Wikipedia, webtext)
  - Configurable text_field per dataset
  - Target-aware caching: re-builds shards when token target changes
"""

import json
import random
import sys
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

# ---------------------------------------------------------------------------
# Dataset mix: ~580M tokens, 55% English / 45% Korean
#
# English (320M):
#   - fineweb_edu:   120M  high-quality educational web text
#   - open_web_math:  80M  math / STEM (critical for thinking tokens)
#   - wikipedia_en:   50M  factual reference
#   - cosmopedia:     40M  synthetic textbook explanations (reasoning-heavy)
#   - finepdfs_edu:   30M  educational PDFs (science, textbooks)
#
# Korean (260M):
#   - korean_webtext_edu:  90M  educational web text
#   - fineweb2_edu_ko:     90M  educational web text
#   - wikipedia_ko:        50M  factual reference
#   - korean_webtext:      30M  general web text
# ---------------------------------------------------------------------------

DATASETS = [
    # --- English (320M tokens, 55%) ---
    {
        "slug": "fineweb_edu_en",
        "path": "HuggingFaceFW/fineweb_edu_100BT",
        "split": "train",
        "target_tokens": 120_000_000,
        "text_field": "text",
    },
    {
        "slug": "open_web_math_en",
        "path": "open-web-math/open-web-math",
        "split": "train",
        "target_tokens": 80_000_000,
        "text_field": "text",
    },
    {
        "slug": "finepdfs_edu_en",
        "path": "HuggingFaceFW/finepdfs_edu_100BT",
        "split": "train",
        "target_tokens": 30_000_000,
        "text_field": "text",
    },
    {
        "slug": "wikipedia_en",
        "path": "wikimedia/wikipedia",
        "name": "20231101.en",
        "split": "train",
        "target_tokens": 50_000_000,
        "text_field": "text",
    },
    {
        "slug": "cosmopedia_en",
        "path": "HuggingFaceTB/cosmopedia",
        "split": "train",
        "target_tokens": 40_000_000,
        "text_field": "text",
    },
    # --- Korean (260M tokens, 45%) ---
    {
        "slug": "korean_webtext_edu",
        "path": "eliceai/korean-webtext-edu",
        "split": "train",
        "target_tokens": 90_000_000,
        "text_field": "text",
    },
    {
        "slug": "fineweb2_edu_ko",
        "path": "minpeter/fineweb-2-edu-korean",
        "split": "train",
        "target_tokens": 90_000_000,
        "text_field": "text",
    },
    {
        "slug": "wikipedia_ko",
        "path": "wikimedia/wikipedia",
        "name": "20231101.ko",
        "split": "train",
        "target_tokens": 50_000_000,
        "text_field": "text",
    },
    {
        "slug": "korean_webtext",
        "path": "HAERAE-HUB/KOREAN-WEBTEXT",
        "split": "train",
        "target_tokens": 30_000_000,
        "text_field": "text",
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
    text_field = spec.get("text_field", "text")
    for row in load_stream(spec):
        text = row.get(text_field)
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


def needs_rebuild(spec, state):
    """Check if a shard needs to be rebuilt (target changed or incomplete)."""
    slug = spec["slug"]
    source_state = state.get(slug, {})
    train_shard, val_shard = shard_paths(spec)

    if not source_state.get("complete"):
        return True
    if not train_shard.exists() or not val_shard.exists():
        return True
    # Rebuild if target changed
    old_target = source_state.get("target_tokens", 0)
    if old_target != spec["target_tokens"]:
        print(f"  {slug}: target changed {old_target:,} → {spec['target_tokens']:,}, rebuilding")
        return True
    return False


def build_source(tokenizer, spec, state):
    slug = spec["slug"]
    label = f"{spec['path']}" + (f" / {spec['name']}" if spec.get("name") else "")
    train_shard, val_shard = shard_paths(spec)

    if not needs_rebuild(spec, state):
        print(f"Skipping {label}; shard already complete with matching target.")
        return state.get(slug, {})

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
                    "target_tokens": target,
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
        "target_tokens": target,
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


def assemble_final_files_shuffled(state):
    """Read all shards, shuffle at document level, write final files.

    This is the critical fix: instead of concatenating English then Korean,
    documents from ALL sources are interleaved randomly so the model sees
    both languages throughout training.
    """
    FINAL_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    FINAL_VAL.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)

    for split_name, final_path in [("train", FINAL_TRAIN), ("val", FINAL_VAL)]:
        print(f"\nAssembling {split_name} file (shuffled)...")
        lines = []
        source_counts = {}

        for spec in DATASETS:
            source_state = state.get(spec["slug"], {})
            if not source_state.get("complete"):
                print(f"  Skipping {spec['slug']}; incomplete.")
                continue

            shard = shard_paths(spec)[0 if split_name == "train" else 1]
            if not shard.exists():
                print(f"  Skipping {spec['slug']}; shard file missing.")
                continue

            count = 0
            with shard.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        lines.append(line)
                        count += 1
            source_counts[spec["slug"]] = count
            print(f"  Loaded {spec['slug']}: {count:,} docs")

        print(f"  Total documents: {len(lines):,}")
        print(f"  Shuffling...")
        rng.shuffle(lines)

        print(f"  Writing to {final_path}...")
        with final_path.open("w", encoding="utf-8") as out:
            for line in lines:
                out.write(line)

        print(f"  Done. {split_name} file: {len(lines):,} documents")

        # Show per-source breakdown
        for slug, count in sorted(source_counts.items()):
            print(f"    {slug}: {count:,} docs")


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

    total = sum(s["target_tokens"] for s in DATASETS)
    en_total = sum(s["target_tokens"] for s in DATASETS if s["slug"].endswith("_en"))
    ko_total = total - en_total

    print(f"\n{'='*60}")
    print(f"Completed sources: {complete}/{len(DATASETS)}")
    print(f"Train tokens written: {train_total:,}")
    print(f"Val tokens written:   {val_total:,}")
    print(f"Target budget:        {total:,} ({en_total:,} EN / {ko_total:,} KO)")
    print(f"EN/KO ratio:          {en_total/total*100:.0f}% / {ko_total/total*100:.0f}%")
    print(f"Final train file:     {FINAL_TRAIN}")
    print(f"Final val file:       {FINAL_VAL}")
    print(f"{'='*60}")


def main():
    ensure_dirs()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    state = load_state()

    total_target = sum(s["target_tokens"] for s in DATASETS)
    print(f"Tokenizer: {TOKENIZER_NAME}")
    print(f"Target token budget: {total_target:,}")
    print(f"Shard directory: {SHARD_DIR}")
    print(f"Number of sources: {len(DATASETS)}")
    print()

    for spec in DATASETS:
        build_source(tokenizer, spec, state)

    assemble_final_files_shuffled(state)
    print_summary(state)


if __name__ == "__main__":
    main()
