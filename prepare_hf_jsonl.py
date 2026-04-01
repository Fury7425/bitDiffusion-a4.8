import json
import random
import sys
import io
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

# --- Hardware/Path Config ---
TOKENIZER_NAME = "Qwen/Qwen-tokenizer"
VAL_RATIO = 0.005 
SEED = 42
MIN_TOKENS = 32
MAX_TOKENS = 32768
SHUFFLE_BUFFER_SIZE = 100_000  # Fits comfortably in 16GB RAM

TRAIN_DIR = Path("data/train")
VAL_DIR = Path("data/val")
SHARD_DIR = Path("data/hf_shards")
STATE_PATH = SHARD_DIR / "progress.json"
FINAL_TRAIN = TRAIN_DIR / "hf_mix_train.jsonl"
FINAL_VAL = VAL_DIR / "hf_mix_val.jsonl"

# --- Balanced 2B Token Mix (Wiki : Math : Reasoning) ---
# Total: ~2.05B Tokens | EN: ~1.1B (54%) | KO: ~0.95B (46%)
DATASETS = [
    # 1. WIKI / FACTUAL (300M)
    {"slug": "wiki_en", "path": "wikimedia/wikipedia", "name": "20231101.en", "target_tokens": 150_000_000},
    {"slug": "wiki_ko", "path": "wikimedia/wikipedia", "name": "20231101.ko", "target_tokens": 150_000_000},
    
    # 2. MATH / LOGIC (700M)
    {"slug": "math_en", "path": "open-web-math/open-web-math", "target_tokens": 400_000_000},
    {"slug": "math_ko", "path": "ChuGyouk/AI-MO-NuminaMath-CoT-Ko", "target_tokens": 300_000_000},
    
    # 3. EDU / REASONING / STUFFS (1.05B)
    {"slug": "fineweb_edu_en", "path": "HuggingFaceFW/fineweb_edu_100BT", "target_tokens": 550_000_000},
    {"slug": "fineweb_edu_ko", "path": "minpeter/fineweb-2-edu-korean", "target_tokens": 500_000_000},
]

def ensure_dirs():
    for d in [TRAIN_DIR, VAL_DIR, SHARD_DIR]: d.mkdir(parents=True, exist_ok=True)

def shard_paths(spec):
    return (SHARD_DIR / f"{spec['slug']}.train.jsonl", SHARD_DIR / f"{spec['slug']}.val.jsonl")

def load_state():
    return json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}

def save_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2))

def format_row(row):
    """Detects and formats math/conversational/standard text."""
    if "text" in row: return row["text"]
    if "problem" in row: # Math CoT format
        return f"Question: {row['problem']}\n\nSolution: <think>\n{row.get('thought','')}\n</think>\n{row.get('solution','')}"
    if "conversations" in row: # Chat format
        return "\n\n".join([f"{m['from']}: {m['value']}" for m in row["conversations"]])
    return None

def build_source(tokenizer, spec, state):
    slug = spec["slug"]
    train_shard, val_shard = shard_paths(spec)
    
    # Target-aware rebuild check
    if state.get(slug, {}).get("complete") and state[slug].get("target_tokens") == spec["target_tokens"]:
        if train_shard.exists(): return state[slug]

    print(f"-> Processing {slug}...")
    rng = random.Random(f"{SEED}:{slug}")
    
    ds_kwargs = {"path": spec["path"], "split": "train", "streaming": True}
    if "name" in spec: ds_kwargs["name"] = spec["name"]
    
    ds = load_dataset(**ds_kwargs)
    written, docs = 0, 0
    
    with train_shard.open("w", encoding="utf-8") as tf, val_shard.open("w", encoding="utf-8") as vf:
        for row in ds:
            text = format_row(row)
            if not text: continue
            
            tokens = len(tokenizer.encode(text, add_special_tokens=False))
            if tokens < MIN_TOKENS or tokens > MAX_TOKENS: continue
            
            line = json.dumps({"text": text.strip()}, ensure_ascii=False) + "\n"
            if rng.random() < VAL_RATIO: vf.write(line)
            else: tf.write(line)
            
            written += tokens
            docs += 1
            if docs % 10000 == 0: print(f"   {slug}: {written/1e6:.1f}M tokens...")
            if written >= spec["target_tokens"]: break

    state[slug] = {"complete": True, "target_tokens": spec["target_tokens"], "tokens": written}
    save_state(state)
    return state[slug]

def assemble_streamed_shuffle(state):
    """Streams shards into a reservoir buffer to shuffle without 16GB RAM limit."""
    rng = random.Random(SEED)
    
    for split in ["train", "val"]:
        final_path = FINAL_TRAIN if split == "train" else FINAL_VAL
        print(f"\nAssembling {split} (Streaming Shuffle)...")
        
        # Open all relevant shards
        shards = []
        for spec in DATASETS:
            p = shard_paths(spec)[0 if split == "train" else 1]
            if p.exists(): shards.append(p.open("r", encoding="utf-8"))
        
        buffer = []
        with final_path.open("w", encoding="utf-8") as out:
            # Fill initial buffer
            print(f"   Filling shuffle reservoir ({SHUFFLE_BUFFER_SIZE} lines)...")
            active_shards = shards[:]
            while len(buffer) < SHUFFLE_BUFFER_SIZE and active_shards:
                for s in active_shards[:]:
                    line = s.readline()
                    if not line: active_shards.remove(s); continue
                    buffer.append(line)
            
            # Stream: Pull one, push one
            print(f"   Streaming to {final_path.name}...")
            rng.shuffle(buffer)
            while active_shards:
                for s in active_shards[:]:
                    line = s.readline()
                    if not line:
                        active_shards.remove(s)
                        continue
                    
                    # Random swap with buffer
                    idx = rng.randint(0, len(buffer)-1)
                    out.write(buffer[idx])
                    buffer[idx] = line
            
            # Flush remaining buffer
            rng.shuffle(buffer)
            for line in buffer: out.write(line)
            
        for s in shards: s.close()

def main():
    ensure_dirs()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    state = load_state()
    for spec in DATASETS: build_source(tokenizer, spec, state)
    assemble_streamed_shuffle(state)
    print("\n[Success] Dataset prepared and streamed.")

if __name__ == "__main__":
    main()