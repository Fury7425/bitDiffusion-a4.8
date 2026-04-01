"""
Prepare shuffled bilingual (EN+KO) JSONL training data for BitDiffusion a4.8.

Upgrades:
 - Memory-safe global shuffle (streamed, not full RAM load)
 - Buffered intra-shard shuffle to prevent sequential bias
 - Same dataset structure and token logic
"""

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

# ===================== DATASETS =====================

DATASETS = [
    # --- English ---
    {"slug": "fineweb_edu_en", "path": "HuggingFaceFW/fineweb_edu_100BT", "split": "train", "target_tokens": 120_000_000, "text_field": "text"},
    {"slug": "open_web_math_en", "path": "open-web-math/open-web-math", "split": "train", "target_tokens": 80_000_000, "text_field": "text"},
    {"slug": "finepdfs_edu_en", "path": "HuggingFaceFW/finepdfs_edu_100BT", "split": "train", "target_tokens": 30_000_000, "text_field": "text"},
    {"slug": "wikipedia_en", "path": "wikimedia/wikipedia", "name": "20231101.en", "split": "train", "target_tokens": 50_000_000, "text_field": "text"},
    {"slug": "cosmopedia_en", "path": "HuggingFaceTB/cosmopedia", "split": "train", "target_tokens": 40_000_000, "text_field": "text"},

    # --- Korean ---
    {"slug": "korean_webtext_edu", "path": "eliceai/korean-webtext-edu", "split": "train", "target_tokens": 90_000_000, "text_field": "text"},
    {"slug": "fineweb2_edu_ko", "path": "minpeter/fineweb-2-edu-korean", "split": "train", "target_tokens": 90_000_000, "text_field": "text"},
    {"slug": "wikipedia_ko", "path": "wikimedia/wikipedia", "name": "20231101.ko", "split": "train", "target_tokens": 50_000_000, "text_field": "text"},
    {"slug": "korean_webtext", "path": "HAERAE-HUB/KOREAN-WEBTEXT", "split": "train", "target_tokens": 30_000_000, "text_field": "text"},
]

# ===================== UTIL =====================

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
    return len(tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False
    )["input_ids"])

def needs_rebuild(spec, state):
    slug = spec["slug"]
    source_state = state.get(slug, {})
    train_shard, val_shard = shard_paths(spec)

    if not source_state.get("complete"):
        return True
    if not train_shard.exists() or not val_shard.exists():
        return True
    if source_state.get("target_tokens") != spec["target_tokens"]:
        return True
    return False

# ===================== BUILD =====================

def build_source(tokenizer, spec, state):
    slug = spec["slug"]
    train_shard, val_shard = shard_paths(spec)

    if not needs_rebuild(spec, state):
        print(f"Skipping {slug}")
        return

    rng = random.Random(f"{SEED}:{slug}")
    target = spec["target_tokens"]

    docs = 0
    written = 0
    train_tokens = 0
    val_tokens = 0

    buffer = []

    print(f"Building {slug} → {target:,} tokens")

    with train_shard.open("w", encoding="utf-8") as train_f, \
         val_shard.open("w", encoding="utf-8") as val_f:

        for text in iter_texts(spec):
            token_count = count_tokens(tokenizer, text)

            if token_count < MIN_TOKENS or token_count > MAX_TOKENS:
                continue

            line = json.dumps({"text": text}, ensure_ascii=False) + "\n"
            buffer.append((line, token_count))

            if len(buffer) >= 2048:
                rng.shuffle(buffer)
                for l, tc in buffer:
                    if rng.random() < VAL_RATIO:
                        val_f.write(l)
                        val_tokens += tc
                    else:
                        train_f.write(l)
                        train_tokens += tc

                    written += tc
                    docs += 1

                buffer.clear()

            if written >= target:
                break

        # flush buffer
        if buffer:
            rng.shuffle(buffer)
            for l, tc in buffer:
                if rng.random() < VAL_RATIO:
                    val_f.write(l)
                    val_tokens += tc
                else:
                    train_f.write(l)
                    train_tokens += tc

                written += tc
                docs += 1

    state[slug] = {
        "complete": True,
        "target_tokens": target,
        "docs": docs,
        "tokens": written,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
    }
    save_state(state)

    print(f"Done {slug}: {written:,} tokens")

# ===================== SHUFFLE =====================

def assemble_final_files_shuffled(state):
    FINAL_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    FINAL_VAL.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    BUFFER_SIZE = 30000  # safe for 16GB RAM

    def stream_shuffled(files):
        buffers = []

        f_handles = [open(f, "r", encoding="utf-8") for f in files]

        try:
            while True:
                for f in f_handles:
                    for _ in range(max(1, BUFFER_SIZE // len(f_handles))):
                        line = f.readline()
                        if line:
                            buffers.append(line)

                if not buffers:
                    break

                rng.shuffle(buffers)

                for line in buffers:
                    yield line

                buffers.clear()

        finally:
            for f in f_handles:
                f.close()

    for split_name, final_path in [("train", FINAL_TRAIN), ("val", FINAL_VAL)]:
        print(f"\nShuffling {split_name}...")

        shard_list = []
        for spec in DATASETS:
            if state.get(spec["slug"], {}).get("complete"):
                shard = shard_paths(spec)[0 if split_name == "train" else 1]
                if shard.exists():
                    shard_list.append(shard)

        with final_path.open("w", encoding="utf-8") as out:
            count = 0
            for line in stream_shuffled(shard_list):
                out.write(line)
                count += 1

                if count % 100000 == 0:
                    print(f"  {count:,} lines")

        print(f"Done {split_name}: {count:,} lines")

# ===================== MAIN =====================

def main():
    ensure_dirs()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    state = load_state()

    for spec in DATASETS:
        build_source(tokenizer, spec, state)

    assemble_final_files_shuffled(state)

if __name__ == "__main__":
    main()