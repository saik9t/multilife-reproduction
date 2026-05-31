"""
sample_subset_sql.py — Stratified sampling from the MultiLifeQA sql/ dataset
to create a small, representative subset for DP evaluation.

Usage:
    python sample_subset_sql.py \
        --data-root ./gen_data_processed/sql \
        --out-root ./gen_data_processed/sql_subset \
        --per-file 5
"""

import argparse
import json
import os
import random

LEAF_KINDS = {"AS", "CQ", "FQ", "NC", "TA"}


def find_jsonl_files(data_root: str):
    """Traverse single_user/ and multi_user/ directories under sql/"""
    found = []
    for root, dirs, files in os.walk(data_root):
        rel = os.path.relpath(root, data_root)
        parts = rel.split(os.sep)
        # 需要至少 scope/bucket/dataset 三层
        if len(parts) < 3:
            continue
        scope = parts[0]  # single_user 或 multi_user
        if scope not in ("single_user", "multi_user"):
            continue
        for fn in files:
            if not fn.lower().endswith(".jsonl"):
                continue
            kind = os.path.splitext(fn)[0].upper()
            if kind not in LEAF_KINDS:
                continue
            fpath = os.path.join(root, fn)
            found.append((fpath, rel, kind, fn))
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
                # sql 格式用 Query_sql + Query_base + Answer
                if "Answer" in obj and ("Query_sql" in obj or "Query" in obj):
                    items.append(obj)
            except Exception:
                pass
    return items


def main():
    ap = argparse.ArgumentParser(description="Stratified sampling from the sql/ dataset")
    ap.add_argument("--data-root", type=str, required=True,
                    help="Path to the original sql/ folder")
    ap.add_argument("--out-root", type=str, required=True,
                    help="Path to the output subset folder")
    ap.add_argument("--per-file", type=int, default=3,
                    help="Maximum samples per JSONL file (default: 3)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--easy-boost", type=int, default=7,
                    help="Additional samples for FQ/CQ in single/")
    args = ap.parse_args()

    random.seed(args.seed)
    files = find_jsonl_files(args.data_root)

    if not files:
        print(f"[ERR] No valid JSONL files found in {args.data_root}")
        return

    total_sampled = 0
    stats_by_type = {}
    stats_by_scope = {}
    stats_by_bucket = {}

    for fpath, rel_dir, kind, fn in files:
        items = read_all_lines(fpath)
        if not items:
            continue

        parts = rel_dir.split(os.sep)
        scope = parts[0]   # single_user / multi_user
        bucket = parts[1] if len(parts) > 1 else ""

        # 同理，单维度简单题多抽，确保有成功案例
        if bucket == "single" and kind in {"FQ", "CQ"}:
            n = min(len(items), args.per_file + args.easy_boost)
        else:
            n = min(len(items), args.per_file)

        sampled = random.sample(items, n)

        out_dir = os.path.join(args.out_root, rel_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, fn)

        with open(out_path, "w", encoding="utf-8") as wf:
            for obj in sampled:
                wf.write(json.dumps(obj, ensure_ascii=False) + "\n")

        total_sampled += n
        stats_by_type[kind] = stats_by_type.get(kind, 0) + n
        stats_by_scope[scope] = stats_by_scope.get(scope, 0) + n
        stats_by_bucket[f"{scope}/{bucket}"] = stats_by_bucket.get(f"{scope}/{bucket}", 0) + n

    print(f"\n=== 采样已经完成 ===")
    print(f"输出目录: {args.out_root}")
    print(f"总采样数: {total_sampled}")
    print(f"\n按题型:")
    for k in sorted(stats_by_type):
        print(f"  {k}: {stats_by_type[k]}")
    print(f"\n按用户范围:")
    for k in sorted(stats_by_scope):
        print(f"  {k}: {stats_by_scope[k]}")
    print(f"\n按维度:")
    for k in sorted(stats_by_bucket):
        print(f"  {k}: {stats_by_bucket[k]}")


if __name__ == "__main__":
    main()