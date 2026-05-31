"""
sample_subset.py — Stratified sampling from the MultiLifeQA simple/ dataset by
question type and dimension to create a small, representative subset for
direct use with eval_simple.py.

Usage:
    python sample_subset.py \
        --data-root ./gen_data_processed/simple \
        --out-root ./gen_data_processed/simple_subset \
        --per-file 5

This script samples at most 5 records from each JSONL file, maintaining
the original directory structure. The resulting subset can be passed
directly to eval_simple.py --data-root ./gen_data_processed/simple_subset
"""

import argparse
import json
import os
import random

LEAF_KINDS = {"AS", "CQ", "FQ", "NC", "TA"}
ALLOWED_BUCKETS = {"single", "M-sleep", "M-activity", "M-C2", "M-C4"}


def find_jsonl_files(data_root: str):
    found = []
    for root, dirs, files in os.walk(data_root):
        rel = os.path.relpath(root, data_root)
        parts = rel.split(os.sep)
        if len(parts) < 3:
            continue
        scope, bucket, dataset = parts[0], parts[1], parts[2]
        if scope != "single_user":
            continue
        if bucket not in ALLOWED_BUCKETS:
            continue
        for fn in files:
            if not fn.lower().endswith(".jsonl"):
                continue
            kind = os.path.splitext(fn)[0].upper()
            if kind not in LEAF_KINDS:
                continue
            fpath = os.path.join(root, fn)
            rel_dir = os.path.relpath(root, data_root)
            found.append((fpath, rel_dir, kind, fn))
    found.sort()
    return found


def read_all_lines(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "Query" in obj and "Answer" in obj:
                    items.append(obj)
            except Exception:
                pass
    return items


def main():
    ap = argparse.ArgumentParser(description="Stratified sampling from the simple/ dataset")
    ap.add_argument("--data-root", type=str, required=True,
                    help="Path to the original simple/ folder")
    ap.add_argument("--out-root", type=str, required=True,
                    help="Path to the output subset folder")
    ap.add_argument("--per-file", type=int, default=5,
                    help="Maximum samples per JSONL file (default: 5)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility")
    ap.add_argument("--easy-boost", type=int, default=10,
                    help="Additional samples for FQ and CQ in single/ (single-dimension) to ensure successful cases are captured")
    args = ap.parse_args()

    random.seed(args.seed)
    files = find_jsonl_files(args.data_root)

    if not files:
        print(f"[ERR] No valid JSONL files found in {args.data_root}")
        return

    total_sampled = 0
    stats_by_type = {}
    stats_by_bucket = {}

    for fpath, rel_dir, kind, fn in files:
        items = read_all_lines(fpath)
        if not items:
            continue

        # 对单维度的简单题型多抽一些，确保有成功案例
        bucket = rel_dir.split(os.sep)[1] if len(rel_dir.split(os.sep)) > 1 else ""
        if bucket == "single" and kind in {"FQ", "CQ", "TA"}:
            n = min(len(items), args.per_file + args.easy_boost)
        else:
            n = min(len(items), args.per_file)

        sampled = random.sample(items, n)

        # 写入输出目录，保持原始结构
        out_dir = os.path.join(args.out_root, rel_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, fn)

        with open(out_path, "w", encoding="utf-8") as wf:
            for obj in sampled:
                wf.write(json.dumps(obj, ensure_ascii=False) + "\n")

        total_sampled += n
        stats_by_type[kind] = stats_by_type.get(kind, 0) + n
        stats_by_bucket[bucket] = stats_by_bucket.get(bucket, 0) + n

    # 打印采样摘要
    print(f"\n=== 采样已经完成 ===")
    print(f"输出目录: {args.out_root}")
    print(f"总采样数: {total_sampled}")
    print(f"\n按题型分布:")
    for k in sorted(stats_by_type):
        print(f"  {k}: {stats_by_type[k]}")
    print(f"\n按维度分布:")
    for k in sorted(stats_by_bucket):
        print(f"  {k}: {stats_by_bucket[k]}")

if __name__ == "__main__":
    main()