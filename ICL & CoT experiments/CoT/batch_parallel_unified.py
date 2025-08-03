import os, json, time, datetime, pathlib, argparse
import pandas as pd
from openai import OpenAI
from sklearn.metrics import f1_score
import numpy as np

from fire import Fire

import sys, dotenv
sys.path.append('../..') 
dotenv.load_dotenv()


DATASET_CONFIGS = {
    "conan": {
        "csv_in": "Tasks/conan/test.csv",
        "csv_out": "outputs/o3-mini/conan/test_passes_parallel.csv",
        "labels": ["facts", "support", "denouncing", "hypocrisy", "unrelated", "humor", "question"],
        "system_msg": "You are an expert in social psychology. Given CONTEXT and COMMENT, reply only with one of: facts, support, denouncing, hypocrisy, unrelated, humor, question",
        "user_format": "context_comment",  # Uses both context and comment
        "label_mapping": None,  # No mapping needed, already strings
        "payload_prefix": "conan"
    },
    "elf22": {
        "csv_in": "Tasks/elf22/test.csv", 
        "csv_out": "outputs/o3-mini/elf22/test_passes_parallel.csv",
        "labels": ["troll", "counterspeech"],
        "system_msg": "You are an expert in social psychology. Given CONTEXT and COMMENT, reply only with one of: troll, counterspeech.",
        "user_format": "context_comment",
        "label_mapping": {1: "troll", 2: "counterspeech"},
        "payload_prefix": "elf22"
    },
    "germeval": {
        "csv_in": "Tasks/germeval/test.csv",
        "csv_out": "outputs/o3-mini/germeval/test_passes_parallel.csv", 
        "labels": ["non-offensive", "offensive"],
        "system_msg": "You are an expert in social psychology. Given a TEXT, reply only with one of: non-offensive, offensive.",
        "user_format": "text_only",  # Only uses text, no context
        "label_mapping": {0: "non-offensive", 1: "offensive"},
        "payload_prefix": "germeval"
    },
    "hscs": {
        "csv_in": "Tasks/hs_cs/test.csv",
        "csv_out": "outputs/o3-mini/hs_cs/test_passes_parallel.csv",
        "labels": ["hatespeech", "counterspeech", "neither"],
        "system_msg": "You are an expert in social psychology. Given CONTEXT and COMMENT, reply only with one of: hatespeech, counterspeech, neither.",
        "user_format": "context_comment",
        "label_mapping": {0: "hatespeech", 1: "counterspeech", 2: "neither"},
        "payload_prefix": "hscs"
    }
}

# ─── GLOBAL CONFIG ──────────────────────────────────────────────────────
MODEL           = "o3-mini"
ITERATIONS      = 4
COMPLETION_WIN  = "24h"        # 25% off = "1h" | 50% off = "24h"
POLL_SECONDS    = 30

def load_and_prep_data(config):
    """Load dataset and apply label mapping if needed."""
    df = (
        pd.read_csv(config["csv_in"])
          .dropna(subset=["text"])
          .reset_index(drop=True)
    )
    
    if config["label_mapping"]:
        df["Class"] = df["Class"].map(config["label_mapping"])
    
    print(f"Loaded {len(df)} rows from {config['csv_in']}")
    return df

def build_user_message(row, user_format):
    """Build user message based on format type."""
    if user_format == "context_comment":
        return f"CONTEXT:\n{row['context']}\n\nCOMMENT:\n{row['text']}\nLABEL:"
    elif user_format == "text_only":
        return f"TEXT:\n{row['text']}\nLABEL:"
    else:
        raise ValueError(f"Unknown user_format: {user_format}")

def build_payload(rows: pd.DataFrame, path: pathlib.Path, config, model):
    """Write NDJSON lines for the batch endpoint."""
    with path.open("w", encoding="utf-8") as fp:
        for idx, r in rows.iterrows():
            messages = [
                {"role": "system", "content": config["system_msg"]},
                {"role": "user", "content": build_user_message(r, config["user_format"])}
            ]
            fp.write(json.dumps({
                "custom_id": str(idx),
                "method": "POST", 
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": messages
                }
            }, ensure_ascii=False) + "\n")

def download_results(client, batch_id: str, pass_id: int, pred_matrix):
    """Download results from completed batch."""
    result_stream = client.files.content(
        client.batches.retrieve(batch_id).output_file_id
    )
    for ln in result_stream.iter_lines():
        if not ln:
            continue
        item    = json.loads(ln)
        row_idx = int(item["custom_id"])
        label   = (
            item["response"]["body"]["choices"][0]["message"]["content"]
              .strip().lower()
        )
        pred_matrix[pass_id][row_idx] = label

def run_experiment(dataset, output_dir):
    """Main experiment logic."""
    config = DATASET_CONFIGS[dataset]
    config["csv_out"] = os.path.join(output_dir, config["csv_out"])
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Load and prepare data
    df = load_and_prep_data(config)
    pred_matrix = {i: {} for i in range(ITERATIONS)}
    
    # ─── 1. CREATE ALL BATCHES ─────────────────────────────────────────
    batch_info = []
    
    for i in range(ITERATIONS):
        payload_path = pathlib.Path(f"{config['payload_prefix']}_payload_pass_{i}.ndjson")
        build_payload(df, payload_path, config, MODEL)
        
        upload = client.files.create(file=open(payload_path, "rb"), purpose="batch")
        batch  = client.batches.create(
            input_file_id     = upload.id,
            endpoint          = "/v1/chat/completions", 
            completion_window = COMPLETION_WIN
        )
        batch_info.append({"id": batch.id, "pass": i, "status": batch.status})
        print(f"PASS {i}: batch {batch.id} → status {batch.status}")
    
    # ─── 2. POLL ALL BATCHES ───────────────────────────────────────────
    print("\n⏳ Polling batches until completion...")
    unfinished = {b["id"]: b for b in batch_info}
    
    while unfinished:
        time.sleep(POLL_SECONDS)
        for bid in list(unfinished):
            b  = client.batches.retrieve(bid)
            rc = b.request_counts or {}
            done  = getattr(rc, "completed", 0)
            total = getattr(rc, "total", "?")
            now   = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"{now} batch {b.id} {b.status:<10} {done}/{total}")
            
            if b.status in {"completed", "failed", "expired"}:
                if b.status != "completed":
                    raise RuntimeError(f"Batch {b.id} ended with {b.status}")
                download_results(client, b.id, unfinished[bid]["pass"], pred_matrix)
                del unfinished[bid]
    
    print("All batches finished")
    
    # ─── 3. MERGE PREDICTIONS & SAVE ───────────────────────────────────
    for i in range(ITERATIONS):
        df[f"pred_pass_{i}"] = df.index.map(lambda idx: pred_matrix[i].get(idx, "unknown"))
    
    # Ensure output directory exists
    pathlib.Path(config["csv_out"]).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config["csv_out"], index=False)
    print(f"Saved: {config['csv_out']}")
    
    # ─── 4. CALCULATE METRICS ──────────────────────────────────────────
    print(f"Results for {dataset}:")
    f1_scores = []

    for i in range(ITERATIONS):
        pred_col = f"pred_pass_{i}"
        unique_preds = df[pred_col].unique()
        print(f"Pass {i} unique predictions: {unique_preds}")
        
        f1 = f1_score(df['Class'], df[pred_col], average='macro')
        f1_scores.append(f1)
        print(f"Pass {i} F1-macro: {f1:.4f}")
    
    print(f"Average F1-macro: {np.mean(f1_scores):.4f}")
    print(f"Std F1-macro: {np.std(f1_scores):.4f}")



def main(dataset : str = None, output_dir: str = None):
    
    print("=" * 70)
    print(f"Parallel Batch Experiment")
    print(f"Dataset: {dataset}")
    print("=" * 70)
    
    
    run_experiment(dataset=dataset, output_dir=output_dir)

if __name__ == "__main__":
    Fire(main)
