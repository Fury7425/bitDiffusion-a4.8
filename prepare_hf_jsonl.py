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

# =========================================================
# UPDATED DATASET MIX (wiki + math boosted, balanced EN/KO)
# =========================================================
DATASETS = [
    # =========================
    # ENGLISH (~55%)
    # =========================
    {
        "slug": "fineweb_edu_en",
        "path": "HuggingFaceFW/fineweb_edu_100BT",
        "split": "train",
        "target_tokens": 100_000_000,
        "text_field": "text",
    },
    {
        "slug": "structured_wiki_en",
        "path": "wikimedia/structured-wikipedia",
        "split": "train",
        "target_tokens": 80_000_000,
        "text_field": "text",
    },
    {
        "slug": "open_web_math_en",
        "path": "open-web-math/open-web-math",
        "split": "train",
        "target_tokens": 70_000_000,
        "text_field": "text",
    },
    {
        "slug": "gsm8k_reasoning_en",
        "path": "notefill/gsm8k-instruction",
        "split": "train",
        "target_tokens": 40_000_000,
        "text_field": "question",  # merged below
    },
    {
        "slug": "wikipedia_en",
        "path": "wikimedia/wikipedia",
        "name": "20231101.en",
        "split": "train",
        "target_tokens": 40_000_000,
        "text_field": "text",
    },
    {
        "slug": "cosmopedia_en",
        "path": "HuggingFaceTB/cosmopedia",
        "split": "train",
        "target_tokens": 30_000_000,
        "text_field": "text",
    },

    # =========================
    # KOREAN (~45%)
    # =========================
    {
        "slug": "korean_webtext_edu",
        "path": "eliceai/korean-webtext-edu",
        "split": "train",
        "target_tokens": 80_000_000,
        "text_field": "text",
    },
    {
        "slug": "fineweb2_edu_ko",
        "path": "minpeter/fineweb-2-edu-korean",
        "split": "train",
        "target_tokens": 80_000_000,
        "text_field": "text",
    },
    {
        "slug": "wiki_ko_clean",
        "path": "lcw99/wikipedia-korean-20221001",
        "split": "train",
        "target_tokens": 40_000_000,
        "text_field": "text",
    },
    {
        "slug": "wiki_qa_ko",
        "path": "lcw99/wikipedia-korean-20240501-1million-qna",
        "split": "train",
        "target_tokens": 30_000_000,
        "text_field": "context",
    },
    {
        "slug": "math_reasoning_ko",
        "path": "Mobiusi/math_ko_reasoning_10K",
        "split": "train",
        "target_tokens": 20_000_000,
        "text_field": "question",  # merged below
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


# =========================================================
# 🔥 IMPROVED TEXT EXTRACTION (reasoning-aware)
# =========================================================
def iter_texts(spec):
    text_field = spec.get("text_field", "text")

    for row in load_stream(spec):
        text = None

        # ---- Custom handling for reasoning datasets ----
        if spec["slug"] == "math_reasoning_ko":
            q = row.get("question", "")
            a = row.get("answer", "")
            e = row.get("explanation", "")
            text = f"문제: {q}\n정답: {a}\n풀이: {e}"

        elif spec["slug"] == "gsm8k_reasoning_en":
            q = row.get("question", "")
            a = row.get("answer", "")
            text = f"Question: {q}\nAnswer: {a}"

        else:
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
    slug = spec["slug"]
    source_state = state.get(slug, {})
    train_shard, val_shard = shard_paths(spec)

    if not source_state.get("complete"):
        return True
    if not train_shard.exists() or not val_shard.exists():
        return True

    if source_state.get("target_tokens", 0) != spec["target_tokens"]:
        print(f"{slug}: target changed → rebuilding")
        return True

    return False


def build_source(tokenizer, spec, state):
    slug = spec["slug"]
    train_shard, val_shard = shard_paths(spec)

    if not needs_rebuild(spec, state):
        print(f"Skipping {slug}")
        return state.get(slug, {})

    rng = random.Random(f"{SEED}:{slug}")
    target = spec["target_tokens"]

    docs = 0
    written = 0
    train_tokens = 0
    val_tokens = 0

    print(f"Starting {slug} ({target:,} tokens)")

    with train_shard.open("w", encoding="utf-8") as train_f, val_shard.open("w", encoding="utf-8") as val_f:
        for text in iter_texts(spec):
            token_count = count_tokens(tokenizer, text)

            if token_count < MIN_TOKENS or token_count > MAX_TOKENS:
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
                print(f"{slug}: docs={docs:,}, tokens={written:,}")

            if written >= target:
                break

    result = {
        "complete": True,
        "target_tokens": target,
        "docs": docs,
        "tokens": written,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
    }

    state[slug] = result
    save_state(state)

    print(f"Finished {slug}: {written:,} tokens")
    return result


# =========================================================
# 🔥 GLOBAL SHUFFLE (CRITICAL FOR MULTITASK RETENTION)
# =========================================================
def assemble_final_files_shuffled(state):
    rng = random.Random(SEED)

    for split_name, final_path in [("train", FINAL_TRAIN), ("val", FINAL_VAL)]:
        print(f"\nAssembling {split_name}...")

        lines = []

        for spec in DATASETS:
            shard = shard_paths(spec)[0 if split_name == "train" else 1]

            if not shard.exists():
                continue

            with shard.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        lines.append(line)

        print(f"Loaded {len(lines):,} docs → shuffling...")
        rng.shuffle(lines)

        with final_path.open("w", encoding="utf-8") as out:
            for line in lines:
                out.write(line)

        print(f"Wrote {len(lines):,} docs → {final_path}")


def main():
    ensure_dirs()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    state = load_state()

    for spec in DATASETS:
        build_source(tokenizer, spec, state)

    assemble_final_files_shuffled(state)


if __name__ == "__main__":
    main()