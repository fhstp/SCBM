import os, json, time, datetime, pathlib
import pandas as pd
from openai import OpenAI
from sklearn.metrics import f1_score
import numpy as np
from fire import Fire

import sys, dotenv
sys.path.append('../..') 
dotenv.load_dotenv()

# ─── DEFAULT CONFIG ──────────────────────────────────────────────────────
MODEL          = "o3-mini"
FOLDS          = 5
COMPLETION_WIN = "24h"
POLL_SECONDS   = 30

LABELS = ["counterspeech", "noncounter"]
SYSTEM_MSG = (
    "You are an expert in social psychology. "
    "Given the TEXT, reply only with one of: counterspeech, noncounter."
)

def build_payload(df: pd.DataFrame, path: pathlib.Path, system_msg: str, model: str):
    """Build NDJSON payload for batch API."""
    with path.open("w", encoding="utf-8") as fp:
        for idx, row in df.iterrows():
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user",
                 "content": f"TEXT: <text_begin> {row['text']} <text_end>\nLABEL:"}
            ]
            fp.write(json.dumps({
                "custom_id": str(idx),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": messages,
                }
            }, ensure_ascii=False) + "\n")

def download_results(client: OpenAI, batch_id: str, df: pd.DataFrame):
    """Download results from completed batch and add predictions to dataframe."""
    stream = client.files.content(
        client.batches.retrieve(batch_id).output_file_id
    )
    preds = {}
    for ln in stream.iter_lines():
        if not ln:
            continue
        item    = json.loads(ln)
        row_idx = int(item["custom_id"])
        label   = (
            item["response"]["body"]["choices"][0]["message"]["content"]
            .strip().lower()
        )
        preds[row_idx] = label
    df["predicted_label"] = df.index.map(lambda i: preds.get(i, "unknown"))

def run_tsnh_experiment(
    data_dir: str = None,
    output_dir: str = None
):
    """
    Run TSNH cross-validation experiment with parallel batch processing.
    
    Args:
        data_dir: Directory containing fold CSV files (test_fold_0.csv, test_fold_1.csv, etc.)
        output_dir: Directory to save results
        payload_prefix: Prefix for payload files
    """
    
    output_dir = os.path.join(output_dir, 'outputs', MODEL, 'tsnh')
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ─── 1. PREPARE ALL PAYLOADS & LAUNCH BATCHES ──────────────────────
    batches = {}

    for fold in range(FOLDS):
        csv_path = f"{data_dir}/test_fold_{fold}.csv"
            
        df = pd.read_csv(csv_path).dropna(subset=["text"]).reset_index(drop=True)
        print(f"Fold {fold}: loaded {len(df)} rows from {csv_path}")

        ndjson_path = pathlib.Path(f"tsnh_payload_fold_{fold}.ndjson")
        build_payload(df, ndjson_path, SYSTEM_MSG, MODEL)

        upload = client.files.create(file=open(ndjson_path, "rb"), purpose="batch")
        batch  = client.batches.create(
            input_file_id     = upload.id,
            endpoint          = "/v1/chat/completions",
            completion_window = COMPLETION_WIN
        )
        batches[batch.id] = {"fold": fold, "df": df}
        print(f"FOLD {fold}: batch {batch.id} → {batch.status}")

    if not batches:
        raise RuntimeError("No valid fold files found!")

    # ─── 2. POLL UNTIL EVERY FOLD COMPLETES ────────────────────────────
    pending = set(batches.keys())

    while pending:
        time.sleep(POLL_SECONDS)
        for bid in list(pending):
            b  = client.batches.retrieve(bid)
            rc = b.request_counts or {}
            done  = getattr(rc, "completed", 0)
            total = getattr(rc, "total", "?")
            fold  = batches[bid]["fold"]
            print(f"fold {fold} batch {b.id} {b.status:<10} {done}/{total}")

            if b.status in {"completed", "failed", "expired"}:
                if b.status != "completed":
                    raise RuntimeError(f"Batch {b.id} (fold {fold}) ended with {b.status}")
                download_results(client, b.id, batches[bid]["df"])
                pending.remove(bid)

    print("All folds finished")

    # ─── 3. SAVE PER-FOLD CSVs ──────────────────────────────────────────
    for bid, meta in batches.items():
        fold = meta["fold"]
        output_path = f"{output_dir}/test_fold_{fold}.csv"
        meta["df"].to_csv(output_path, index=False)

    # ─── 4. CALCULATE CROSS-VALIDATION METRICS ─────────────────────────
    f1_scores = []
    
    for bid, meta in batches.items():
        fold = meta["fold"]
        df = meta["df"]
            
        unique_preds = df['predicted_label'].unique()
        print(f"Fold {fold} unique predictions: {unique_preds}")
        
        f1 = f1_score(df['Class'], df['predicted_label'], average='macro')
        f1_scores.append(f1)
        print(f"Fold {fold} F1-macro: {f1:.4f}")
    
    print(f"\nCross-Validation Results:")
    print(f"Average F1-macro: {np.mean(f1_scores):.4f}")
    print(f"Std F1-macro: {np.std(f1_scores):.4f}")
    print(f"F1 scores per fold: {[f'{f:.4f}' for f in f1_scores]}")

def main(
    data_dir: str = "Tasks/tsnh/dataset_5folding",
    output_dir: str = None
):
    
    run_tsnh_experiment(
            data_dir=data_dir,
            output_dir=output_dir
        )

if __name__ == "__main__":
    Fire(main)
