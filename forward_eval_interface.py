#!/usr/bin/env python3
"""
Forward-facing benchmark evaluation interface (intentionally incomplete).

This module is the **contract + scaffold** for future work: load LROBench queries from
metadata, plug in your system via ``your_system_run``, and compare with ``score``.
It is not a finished runner until you implement ``your_system_run``.

Besides ``main``, the intended structure is three pieces:

- ``build_query`` — load metadata rows into ``BenchQuery`` objects
- ``your_system_run`` — empty stub; you implement loading DBs, models, operators, etc.
- ``score`` — compare prediction to ground truth for a given task

Typical prediction shapes for ``score`` (match the existing eval/*.py pipelines):

- **select / match**: ``list[list[Any]]`` or a DataFrame aligned with ``ground_truth`` rows
- **order**: ``list[Any]`` (ordered Top-k)
- **cluster**: ``list[int]``, same length as ``ground_truth``
- **impute**: ``list[Any]`` when metadata ``ground_truth`` is a list (row-aligned column fill)
- **multi_lro**: always exact table match. Scalars are treated as a **1×1** table; nested lists
  are rows×columns; a flat list is an **n×1** table (one column, ordered rows).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.metrics.classic import (  # noqa: E402
    ari,
    f1_score,
    kendall_tau_at_k,
    list_em_acc,
    nmi,
)

from src.metrics.agent_match import list_agent_acc  # noqa: E402

TaskName = str

DEFAULT_METADATA: Dict[TaskName, str] = {
    "multi_lro": os.path.join(PROJECT_ROOT, "multi_metadata.json"),
    "select": os.path.join(PROJECT_ROOT, "select_metadata.json"),
    "match": os.path.join(PROJECT_ROOT, "match_metadata.json"),
    "impute": os.path.join(PROJECT_ROOT, "impute_metadata.json"),
    "cluster": os.path.join(PROJECT_ROOT, "cluster_metadata.json"),
    "order": os.path.join(PROJECT_ROOT, "order_metadata.json"),
}

ALL_TASKS = tuple(DEFAULT_METADATA.keys())


@dataclass
class BenchQuery:
    task: TaskName
    query_id: str
    question: str
    db_used: str
    ground_truth: Any
    attributes: Dict[str, Any]


def build_query(task: TaskName, metadata_path: Optional[str] = None) -> List[BenchQuery]:
    """Load all benchmark queries for one task from the corresponding metadata JSON."""
    path = metadata_path or DEFAULT_METADATA[task]
    with open(path, "r", encoding="utf-8") as f:
        rows = list(json.load(f).get("queries") or [])
    out: List[BenchQuery] = []
    for row in rows:
        out.append(
            BenchQuery(
                task=task,
                query_id=str(row["query_id"]).strip(),
                question=row.get("question", ""),
                db_used=row.get("db_used", ""),
                ground_truth=row.get("ground_truth"),
                attributes=dict(row.get("attributes") or {}),
            )
        )
    return out


def your_system_run(query: BenchQuery) -> Any:
    """
    Replace this with your pipeline: read ./databases/<db_used>/, call LLM / operators, etc.

    Return a prediction in the shape expected by ``score`` for ``query.task`` (see module docstring
    or eval/*.py pipelines). Default ``None`` means “no prediction”.
    """
    return None


def score(task: TaskName, ground_truth: Any, prediction: Any, attributes: Dict[str, Any]) -> Dict[str, Any]:
    """Score one prediction. ``ground_truth`` / ``prediction`` follow metadata + task conventions."""
    gt, pred = ground_truth, prediction

    if task in ("select", "match"):
        if isinstance(gt, pd.DataFrame):
            truth = gt.copy()
        elif isinstance(gt, list):
            truth = pd.DataFrame(gt) if gt else pd.DataFrame()
        else:
            raise TypeError(f"expected list or DataFrame for ground_truth, got {type(gt)}")
        if isinstance(pred, pd.DataFrame):
            pred_df = pred.copy()
        elif isinstance(pred, list):
            pred_df = pd.DataFrame(pred) if pred else pd.DataFrame()
        else:
            raise TypeError(f"expected list or DataFrame for prediction, got {type(pred)}")
        p, r, f1 = f1_score(truth, pred_df)
        return {"metric": "precision, recall, f1", "precision": p, "recall": r, "f1": f1}

    if task == "order":
        if not isinstance(gt, list):
            raise TypeError("order ground_truth must be a list")
        pred_list = pred.tolist() if isinstance(pred, pd.Series) else list(pred)
        hr_at_k, _, _ = f1_score(list(gt), pred_list)
        return {
            "metric": "hr_at_k, tau",
            "hr_at_k": hr_at_k,
            "tau": kendall_tau_at_k(list(gt), pred_list),
        }

    if task == "cluster":
        if not isinstance(gt, list):
            raise TypeError("cluster ground_truth must be a list[int]")
        pred_list = pred.tolist() if isinstance(pred, pd.Series) else list(pred)
        if len(pred_list) != len(gt):
            return {
                "metric": "ari, nmi",
                "error": "length_mismatch",
                "len_truth": len(gt),
                "len_pred": len(pred_list),
                "ari": 0.0,
                "nmi": 0.0,
            }
        return {"metric": "ari, nmi", "ari": ari(gt, pred_list), "nmi": nmi(gt, pred_list)}

    if task == "impute":
        if isinstance(gt, list):
            pred_list = pred.tolist() if isinstance(pred, pd.Series) else list(pred)
            if len(pred_list) != len(gt):
                return {
                    "metric": "list_em_acc",
                    "error": "length_mismatch",
                    "len_truth": len(gt),
                    "len_pred": len(pred_list),
                    "list_em_acc": 0.0,
                }
            return {
                "metric": "em_acc, agent_match_acc",
                "em_acc": list_em_acc(gt, pred_list),
                "agent_match_acc": list_agent_acc(gt, pred_list),
            }
        return {
            "metric": "unsupported",
            "skipped": True,
            "reason": "only list ground_truth via list_em_acc; use df_em_acc + missing_log for cell-level",
        }

    if task == "multi_lro":

        def _to_table(x: Any) -> pd.DataFrame:
            if isinstance(x, pd.DataFrame):
                return x.reset_index(drop=True).copy()
            if isinstance(x, pd.Series):
                return x.to_frame().reset_index(drop=True)
            if isinstance(x, list):
                if not x:
                    return pd.DataFrame()
                if isinstance(x[0], (list, tuple)):
                    return pd.DataFrame(x).reset_index(drop=True)
                return pd.DataFrame([[v] for v in x]).reset_index(drop=True)
            return pd.DataFrame([[x]]).reset_index(drop=True)

        truth = _to_table(gt)
        pred_df = _to_table(pred)
        if truth.shape != pred_df.shape:
            return {"metric": "exact_table", "exact_table": 0.0, "match": False}
        ta = truth.astype(str).fillna("<na>")
        tb = pred_df.astype(str).fillna("<na>")
        match = bool(ta.equals(tb))
        return {"metric": "exact_table", "exact_table": 1.0 if match else 0.0, "match": match}

    return {"skipped": True, "reason": f"unknown task {task!r}"}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "LROBench forward eval interface: implement your_system_run, then run this scaffold "
            "to score against metadata ground truth."
        )
    )
    parser.add_argument(
        "--tasks",
        default="all",
        help=f"Comma-separated task names, or 'all'. Choices: {', '.join(ALL_TASKS)}",
    )
    parser.add_argument("--out", default=None, help="Write full report JSON to this path")
    args = parser.parse_args(argv)

    if args.tasks.strip().lower() == "all":
        tasks: Tuple[str, ...] = ALL_TASKS
    else:
        tasks = tuple(t.strip() for t in args.tasks.split(",") if t.strip())
        bad = [t for t in tasks if t not in DEFAULT_METADATA]
        if bad:
            parser.error(f"unknown task(s): {bad}; valid: {list(ALL_TASKS)}")

    report: Dict[str, Any] = {"tasks": {}}
    for task in tasks:
        queries = build_query(task)
        per_query: List[Dict[str, Any]] = []
        for q in queries:
            pred = your_system_run(q)
            record: Dict[str, Any] = {
                "query_id": q.query_id,
                "has_prediction": pred is not None,
            }
            if pred is None:
                record["scores"] = {"skipped": True, "reason": "your_system_run returned None"}
                per_query.append(record)
                continue
            if q.ground_truth is None:
                record["scores"] = {"skipped": True, "reason": "ground_truth is null"}
                per_query.append(record)
                continue
            try:
                record["scores"] = score(task, q.ground_truth, pred, q.attributes)
            except Exception as e:  # noqa: BLE001
                record["scores"] = {"error": str(e), "error_type": type(e).__name__}
            per_query.append(record)

        scored_rows = [
            r
            for r in per_query
            if not (r.get("scores") or {}).get("skipped") and "error" not in (r.get("scores") or {})
        ]
        summary: Dict[str, Any] = {
            "n_queries": len(per_query),
            "n_scored": len(scored_rows),
            "n_skipped": len(per_query) - len(scored_rows),
        }
        for key in (
            "f1",
            "precision",
            "recall",
            "ari",
            "nmi",
            "kendall_tau_at_k",
            "exact_match",
            "exact_table",
            "list_em_acc",
            "hr_at_k",
            "tau",
        ):
            vals = [
                float(r["scores"][key])
                for r in scored_rows
                if isinstance((r.get("scores") or {}).get(key), (int, float))
            ]
            if vals:
                summary[f"mean_{key}"] = round(sum(vals) / len(vals), 6)

        report["tasks"][task] = {
            "task": task,
            "metadata": DEFAULT_METADATA[task],
            "per_query": per_query,
            "summary": summary,
        }

    summary_lines = []
    for task, body in report["tasks"].items():
        s = body["summary"]
        summary_lines.append(
            f"{task}: queries={s['n_queries']} scored={s['n_scored']} skipped={s['n_skipped']} "
            + " ".join(f"{k}={v}" for k, v in s.items() if k.startswith("mean_") and v is not None)
        )
    print("\n".join(summary_lines))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Wrote: {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
