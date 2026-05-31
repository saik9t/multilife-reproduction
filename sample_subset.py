"""
sample_subset.py — 从 MultiLifeQA simple/ 数据集中按题型和维度分层采样，
生成一个小型代表性子集，供 eval_simple.py 直接使用。

用法:
    python sample_subset.py \
        --data-root ./gen_data_processed/simple \
        --out-root ./gen_data_processed/simple_subset \
        --per-file 5

这会从每个 JSONL 文件中最多抽 5 条，生成的子集保持原始目录结构，
可以直接传给 eval_simple.py --data-root ./gen_data_processed/simple_subset
"""

import argparse
import json
import os
import random

LEAF_KINDS = {"AS", "CQ", "FQ", "NC", "TA"}
ALLOWED_BUCKETS = {"single", "M-sleep", "M-activity", "M-C2", "M-C4"}


def find_jsonl_files(data_root: str):
    """复用 eval_simple.py 的目录遍历逻辑"""
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
    ap = argparse.ArgumentParser(description="从 simple/ 数据集中分层采样")
    ap.add_argument("--data-root", type=str, required=True,
                    help="原始 simple/ 文件夹路径")
    ap.add_argument("--out-root", type=str, required=True,
                    help="输出子集的文件夹路径")
    ap.add_argument("--per-file", type=int, default=5,
                    help="每个 JSONL 文件最多抽多少条 (默认 5)")
    ap.add_argument("--seed", type=int, default=42,
                    help="随机种子，确保可复现")
    ap.add_argument("--easy-boost", type=int, default=10,
                    help="对 single/ (单维度) 下的 FQ 和 CQ 额外多抽几条，确保能看到成功案例")
    args = ap.parse_args()

    random.seed(args.seed)
    files = find_jsonl_files(args.data_root)

    if not files:
        print(f"[ERR] 在 {args.data_root} 下没有找到合法的 JSONL 文件")
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
    print(f"\n=== 采样完成 ===")
    print(f"输出目录: {args.out_root}")
    print(f"总采样数: {total_sampled}")
    print(f"\n按题型分布:")
    for k in sorted(stats_by_type):
        print(f"  {k}: {stats_by_type[k]}")
    print(f"\n按维度分布:")
    for k in sorted(stats_by_bucket):
        print(f"  {k}: {stats_by_bucket[k]}")

    print(f"\n现在可以运行:")
    print(f"  python eval_simple.py \\")
    print(f"    --data-root {args.out_root} \\")
    print(f"    --eval-root ./eval \\")
    print(f"    --model gpt-4o-mini \\")
    print(f"    --max-new-tokens 32 \\")
    print(f"    --api-key \"你的key\"")


if __name__ == "__main__":
    main()