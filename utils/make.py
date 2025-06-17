import pandas as pd
import json
from pathlib import Path
import os

# 데이터 파일 경로
base_dir = Path("data")
json_path = base_dir / "jailbreaks.json"
csv_files = [
    base_dir / "jailbreak_prompts.csv",
    base_dir / "malicous_deepset.csv",
]
large_csv_files = [
    base_dir / "forbidden_question_set_with_prompts.csv",
    base_dir / "forbidden_question_set_df.csv",
    base_dir / "predictionguard_df.csv",
]
parquet_files = [
    base_dir / "jailbreak_prompts.parquet",
]

output_path = base_dir / "combined.jsonl"

jsonl_data = []
unique_texts = set()

# 1. JSON 파일 처리
if json_path.exists():
    with open(json_path, "r", encoding="utf-8") as f:
        jailbreak_data = json.load(f)
    for prompt in jailbreak_data.get("jailbreak", []):
        if prompt not in unique_texts:
            jsonl_data.append({"text": prompt})
            unique_texts.add(prompt)

# 2. 일반 CSV 파일 처리
for csv_path in csv_files:
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, usecols=["Prompt"])
            for prompt in df["Prompt"].dropna():
                if str(prompt) not in unique_texts:
                    jsonl_data.append({"text": str(prompt)})
                    unique_texts.add(str(prompt))
        except Exception as e:
            print(f"[WARN] {csv_path} 처리 중 오류: {e}")

# 3. 대용량 CSV 파일 처리 (chunk 단위)
def process_large_csv(path):
    if not path.exists():
        return
    try:
        for chunk in pd.read_csv(path, usecols=["Prompt"], chunksize=10000):
            for prompt in chunk["Prompt"].dropna():
                if str(prompt) not in unique_texts:
                    jsonl_data.append({"text": str(prompt)})
                    unique_texts.add(str(prompt))
    except Exception as e:
        print(f"[WARN] {path} 처리 중 오류: {e}")

for large_csv in large_csv_files:
    process_large_csv(large_csv)

# 4. Parquet 파일 처리
def process_parquet_file(path):
    if not path.exists():
        return
    try:
        df = pd.read_parquet(path, columns=["user_input"])
        for prompt in df["user_input"].dropna():
            if str(prompt) not in unique_texts:
                jsonl_data.append({"text": str(prompt)})
                unique_texts.add(str(prompt))
    except Exception as e:
        print(f"[WARN] {path} 처리 중 오류: {e}")

for pq_file in parquet_files:
    process_parquet_file(pq_file)

# 5. combined.jsonl로 저장 (중복 제거된 상태)
with open(output_path, "w", encoding="utf-8") as f:
    for entry in jsonl_data:
        json.dump(entry, f, ensure_ascii=False)
        f.write("\n")

print(f"Saved combined dataset to {output_path} (unique count: {len(jsonl_data)})")
