import pandas as pd 
import numpy as np

def append_result(csv_path, pipeline, results, tokens):
    with open(csv_path, 'a', encoding='utf-8') as f:
        f.write(f"{pipeline},")
        for result in results:
            f.write(f"{result},")
        for token in tokens:
            f.write(f"{token},")
        f.write(f"\n")
        f.flush()
    print(f"pipeline_{pipeline}: {results}, {tokens}")


def args_to_filename(args, suffix=".txt"):
    parts = [
        args.operator.value,
        args.impl.value,
        str(args.start),
        str(args.end)
    ]

    if hasattr(args, 'example_num') and args.example_num is not None:
        parts.append(str(args.example_num)+"shot")

    if hasattr(args, 'sort_algo') and args.sort_algo is not None:       
        parts.append(str(args.sort_algo))

    if args.thinking:
        parts.append('thinking')

    filename = "_".join(parts) + suffix
    return filename

def mean_by_position(runs):
    if not runs:
        return ["ERROR"], ["ERROR"]

    # 必须保证每次返回长度一致（都为2或都为3），否则你需要定义对齐规则
    L = len(runs[0])
    if any(len(x) != L for x in runs):
        raise ValueError(f"Inconsistent tuple length in runs: {[len(x) for x in runs]}")

    out = []
    for i in range(L):
        vals = [x[i] for x in runs]
        # 尝试转成数值求平均；若不是数值则无法平均
        try:
            vals_num = [float(v) for v in vals]
            out.append(sum(vals_num) / len(vals_num))
        except Exception:
            # 如果不是数值，改成多数投票/取第一个等策略；这里给出“多数投票”
            from collections import Counter
            out.append(Counter(vals).most_common(1)[0][0])
    return out