import json
import random
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

OUT_TRAIN = Path("data/train/hf_mix_train.jsonl")
OUT_VAL = Path("data/val/hf_mix_val.jsonl")
OUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)
OUT_VAL.parent.mkdir(parents=True, exist_ok=True)

TOKENIZER_NAME = "Qwen/Qwen-tokenizer"
VAL_RATIO = 0.01
SEED = 42
MIN_TOKENS = 32

random.seed(SEED)

# Total target: 580M tokens
DATASETS = [
    {"path": "HuggingFaceFW/fineweb-2", "name": "kor_Hang", "split": "train", "target_tokens": 60_000_000},
    {"path": "HuggingFaceFW/fineweb-2", "name": "jpn_Jpan", "split": "train", "target_tokens": 60_000_000},
    {"path": "HuggingFaceFW/fineweb-2", "name": "zho_Hans", "split": "train", "target_tokens": 60_000_000},
    {"path": "HuggingFaceFW/fineweb-2", "name": "spa_Latn", "split": "train", "target_tokens": 40_000_000},
    {"path": "HuggingFaceFW/fineweb-2", "name": "fra_Latn", "split": "train", "target_tokens": 30_000_000},
    {"path": "HuggingFaceFW/fineweb-2", "name": "deu_Latn", "split": "train", "target_tokens": 30_000_000},
    {"path": "HuggingFaceFW/fineweb-2", "name": "ara_Arab", "split": "train", "target_tokens": 20_000_000},
    {"path": "HuggingFaceFW/fineweb-2", "name": "rus_Cyrl", "split": "train", "target_tokens": 20_000_000},
    {"path": "wikimedia/wikipedia", "name": "20231101.en", "split": "train", "target_tokens": 120_000_000},
    {"path": "wikimedia/wikipedia", "name": "20231101.ko", "split": "train", "target_tokens": 30_000_000},
    {"path": "wikimedia/wikipedia", "name": "20231101.ja", "split": "train", "target_tokens": 20_000_000},
    {"path": "wikimedia/wikipedia", "name": "20231101.zh", "split": "train", "target_tokens": 20_000_000},
    {"path": "wikimedia/wikipedia", "name": "20231101.es", "split": "train", "target_tokens": 10_000_000},
    {"path": "uonlp/CulturaX", "name": "en", "split": "train", "target_tokens": 40_000_000},
    {"path": "uonlp/CulturaX", "name": "ko", "split": "train", "target_tokens": 15_000_000},
    {"path": "uonlp/CulturaX", "name": "ja", "split": "train", "target_tokens": 10_000_000},
    {"path": "uonlp/CulturaX", "name": "zh", "split": "train", "target_tokens": 10_000_000},
    {"path": "uonlp/CulturaX", "name": "es", "split": "train", "target_tokens": 5_000_000},
]


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
        text = row.get("text", "")
        if isinstance(text, str):
            text = text.strip()
        if text:
            yield text


def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    train_tokens = 0
    val_tokens = 0

    print(f"Tokenizer: {TOKENIZER_NAME}")
    print(f"Output train file: {OUT_TRAIN}")
    print(f"Output val file:   {OUT_VAL}")

    with OUT_TRAIN.open("w", encoding="utf-8") as train_f, OUT_VAL.open("w", encoding="utf-8") as val_f:
        for spec in DATASETS:
            target = spec["target_tokens"]
            written = 0
            docs = 0
            label = f"{spec['path']} / {spec['name']}"

            print(f"Starting {label} with target {target:,} tokens")

            for text in iter_texts(spec):
                token_count = len(tokenizer.encode(text, add_special_tokens=False))
                if token_count < MIN_TOKENS:
                    continue

                record = {"text": text}
                line = json.dumps(record, ensure_ascii=False) + "\n"

                if random.random() < VAL_RATIO:
                    val_f.write(line)
                    val_tokens += token_count
                else:
                    train_f.write(line)
                    train_tokens += token_count

                written += token_count
                docs += 1

                if docs % 1000 == 0:
                    print(f"  {label}: docs={docs:,}, tokens={written:,}")

                if written >= target:
                    break

            print(f"Finished {label}: docs={docs:,}, tokens={written:,}")

    print(f"Done. train_tokens={train_tokens:,}, val_tokens={val_tokens:,}")


if __name__ == "__main__":
    main()
