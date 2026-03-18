#!/usr/bin/env python3
"""
analyse.py — Benchmark Results Analyser

Reads the summary JSON produced by benchmark.py and generates:
  - Console report with per-model and per-category breakdowns
  - ASR and MR comparison tables
  - Quantization effect table (FP16 → INT8 → INT4 per family)
  - CSV exports for statistical analysis in R / SPSS / Excel
  - Optional Markdown report (--markdown) for dissertation use

Usage:
  python3 analyse.py results/summary_YYYYMMDD_HHMMSS.json
  python3 analyse.py results/summary_YYYYMMDD_HHMMSS.json --markdown
  python3 analyse.py results/summary_YYYYMMDD_HHMMSS.json --csv-only
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────

def bar(value: float, width: int = 30) -> str:
    """ASCII progress bar for 0.0–1.0 values."""
    filled = int(value * width)
    return "█" * filled + "░" * (width - filled)

def pct(value: float) -> str:
    return f"{value * 100:5.1f}%"

def load_summary(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Main analysis ──────────────────────────────────────────────────────────────

def analyse(summary: dict, output_dir: Path, write_md: bool) -> None:
    run_id    = summary["run_id"]
    summaries = summary["summaries"]

    print(f"\n{'='*70}")
    print(f"  PROMPT INJECTION BENCHMARK — RESULTS ANALYSIS")
    print(f"  Run ID: {run_id}")
    print(f"{'='*70}\n")

    # ── Group data ─────────────────────────────────────────────────────────────
    # by_model[display_name + quant] → list of AttackSummary dicts
    by_model  = defaultdict(list)
    by_attack = defaultdict(list)
    by_cat    = defaultdict(list)

    for s in summaries:
        key = f"{s['display_name']} [{s['quant_label']}]"
        by_model[key].append(s)
        by_attack[s["attack_id"]].append(s)
        by_cat[s["attack_category"]].append(s)

    # ── 1. Per-model summary ───────────────────────────────────────────────────
    print("1. OVERALL ATTACK SUCCESS RATE (ASR) BY MODEL VARIANT")
    print("─" * 70)
    print(f"  {'Model variant':<40} {'ASR':>6}  {'MR':>6}  Bar")
    print(f"  {'─'*40} {'─'*6}  {'─'*6}  {'─'*30}")

    model_rows = []
    for key in sorted(by_model.keys()):
        entries   = by_model[key]
        mean_asr  = sum(e["asr_rate"] for e in entries) / len(entries)
        mean_mr   = sum(e["mr_mean"]  for e in entries) / len(entries)
        model_rows.append((key, mean_asr, mean_mr))
        print(f"  {key:<40} {pct(mean_asr)}  {pct(mean_mr)}  {bar(mean_asr, 25)}")

    # ── 2. Per-category summary ────────────────────────────────────────────────
    print(f"\n2. ATTACK SUCCESS RATE BY CATEGORY")
    print("─" * 70)
    cat_labels = {
        "NI": "Naive Injection       (Liu et al. A1)",
        "EI": "Escape/Format Inject  (Liu et al. A2+A3)",
        "CI": "Context Injection     (Liu et al. A4 adapted)",
        "PL": "Prompt Leaking        (novel)",
        "ML": "Multilingual Attacks  (novel)",
    }
    for cat, label in cat_labels.items():
        entries = by_cat.get(cat, [])
        if not entries:
            continue
        mean_asr = sum(e["asr_rate"] for e in entries) / len(entries)
        mean_mr  = sum(e["mr_mean"]  for e in entries) / len(entries)
        print(f"  {label:<45} ASR={pct(mean_asr)}  MR={pct(mean_mr)}  {bar(mean_asr, 20)}")

    # ── 3. Quantization effect table ──────────────────────────────────────────
    print(f"\n3. QUANTIZATION EFFECT (mean ASR per family per quant level)")
    print("─" * 70)
    print(f"  {'Family':<22} {'FP16':>8} {'INT8':>8} {'INT4':>8}  {'Trend'}")
    print(f"  {'─'*22} {'─'*8} {'─'*8} {'─'*8}  {'─'*20}")

    families = sorted({s["family"] for s in summaries})
    quant_rows = []
    for fam in families:
        row = {}
        for s in summaries:
            if s["family"] != fam:
                continue
            q = s["quant_label"]
            row.setdefault(q, []).append(s["asr_rate"])
        avgs = {q: sum(v)/len(v) for q, v in row.items()}
        fp16 = avgs.get("FP16", None)
        int8 = avgs.get("INT8", None)
        int4 = avgs.get("INT4", None)

        # Trend: does quantization increase susceptibility?
        if fp16 is not None and int4 is not None:
            diff = int4 - fp16
            trend = f"INT4 {'↑' if diff > 0.02 else '↓' if diff < -0.02 else '≈'} FP16 ({diff:+.2f})"
        else:
            trend = "insufficient data"

        fp16_s = pct(fp16) if fp16 is not None else "  N/A "
        int8_s = pct(int8) if int8 is not None else "  N/A "
        int4_s = pct(int4) if int4 is not None else "  N/A "
        print(f"  {fam:<22} {fp16_s:>8} {int8_s:>8} {int4_s:>8}  {trend}")
        quant_rows.append({"family": fam, "FP16": fp16, "INT8": int8, "INT4": int4, "trend": trend})

    # ── 4. Attack grid ─────────────────────────────────────────────────────────
    print(f"\n4. ATTACK × MODEL GRID (ASR)")
    print("─" * 70)
    attack_ids = sorted({s["attack_id"] for s in summaries})
    model_keys = sorted({f"{s['display_name']} [{s['quant_label']}]" for s in summaries})

    # Build lookup: (model_key, attack_id) → asr_rate
    grid = {}
    for s in summaries:
        mkey = f"{s['display_name']} [{s['quant_label']}]"
        grid[(mkey, s["attack_id"])] = s["asr_rate"]

    # Print header
    header = f"  {'Model':<35} " + " ".join(f"{a:>6}" for a in attack_ids)
    print(header)
    print("  " + "─" * (35 + 7 * len(attack_ids)))
    for mkey in model_keys:
        row_str = f"  {mkey:<35} "
        for aid in attack_ids:
            v = grid.get((mkey, aid), None)
            row_str += f"{pct(v):>6} " if v is not None else f"{'N/A':>6} "
        print(row_str)

    # ── 5. ASR vs MR gap ──────────────────────────────────────────────────────
    print(f"\n5. ASR vs MR GAP (gap = MR - ASR, positive = MR catches more)")
    print("─" * 70)
    print("  Attacks where the gap is largest (model complied in unexpected phrasing):")
    gaps = []
    for s in summaries:
        gap_val = s["mr_mean"] - s["asr_rate"]
        if abs(gap_val) > 0.05:
            mkey = f"{s['display_name']} [{s['quant_label']}]"
            gaps.append((gap_val, mkey, s["attack_id"], s["attack_name"]))
    gaps.sort(reverse=True)
    for gap_val, mkey, aid, aname in gaps[:15]:
        print(f"  {gap_val:+.2f}  {mkey:<40} {aid}  {aname}")
    if not gaps:
        print("  No significant gaps found (ASR and MR are closely aligned).")

    # ── Write CSV exports ──────────────────────────────────────────────────────
    # Attack grid CSV
    grid_path = output_dir / f"attack_grid_{run_id}.csv"
    with open(grid_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model_variant"] + attack_ids)
        for mkey in model_keys:
            row = [mkey] + [grid.get((mkey, aid), "") for aid in attack_ids]
            w.writerow(row)

    # Quantization summary CSV
    quant_path = output_dir / f"quant_summary_{run_id}.csv"
    with open(quant_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["family","FP16","INT8","INT4","trend"])
        w.writeheader()
        w.writerows(quant_rows)

    # Full summary CSV (all rows)
    full_path = output_dir / f"summary_{run_id}.csv"
    if summaries:
        with open(full_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=summaries[0].keys())
            w.writeheader()
            w.writerows(summaries)

    print(f"\n  Exports written:")
    print(f"    {grid_path}")
    print(f"    {quant_path}")
    print(f"    {full_path}")

    # ── Write Markdown report ──────────────────────────────────────────────────
    if write_md:
        md_path = output_dir / f"report_{run_id}.md"
        write_markdown(summary, model_rows, quant_rows, grid, model_keys,
                       attack_ids, run_id, md_path)
        print(f"    {md_path}")

    print(f"\n{'='*70}\n")


def write_markdown(summary, model_rows, quant_rows, grid, model_keys,
                   attack_ids, run_id, path):
    lines = [
        f"# Prompt Injection Benchmark Results",
        f"",
        f"**Run ID:** `{run_id}`  ",
        f"**Generated:** {summary['generated_at']}  ",
        f"**Ollama version:** {summary.get('ollama_version', 'unknown')}  ",
        f"",
        f"## Methodology",
        f"",
        f"This benchmark evaluates LLM resistance to prompt injection attacks across",
        f"quantization levels (FP16 → INT8 → INT4), inspired by Liu et al. (2024).",
        f"",
        f"**Metrics:**",
        f"- **ASR** (Attack Success Rate): keyword-matching, binary per trial",
        f"- **MR** (Matched Rate): LLM-as-judge, 0.0/0.5/1.0 per trial",
        f"",
        f"**Experimental constants:** temperature={summary['experimental_constants']['temperature']}, "
        f"seed={summary['experimental_constants']['seed']}, "
        f"context={summary['experimental_constants']['context_length']}, "
        f"max_tokens={summary['experimental_constants']['max_tokens']}",
        f"",
        f"## Results by Model Variant",
        f"",
        f"| Model Variant | ASR | MR |",
        f"|---|---|---|",
    ]
    for key, asr, mr in model_rows:
        lines.append(f"| {key} | {pct(asr)} | {pct(mr)} |")

    lines += [
        f"",
        f"## Quantization Effect",
        f"",
        f"| Family | FP16 | INT8 | INT4 | Trend |",
        f"|---|---|---|---|---|",
    ]
    for r in quant_rows:
        fp = pct(r["FP16"]) if r["FP16"] is not None else "N/A"
        i8 = pct(r["INT8"]) if r["INT8"] is not None else "N/A"
        i4 = pct(r["INT4"]) if r["INT4"] is not None else "N/A"
        lines.append(f"| {r['family']} | {fp} | {i8} | {i4} | {r['trend']} |")

    lines += [
        f"",
        f"## References",
        f"",
        f"- Liu, Y. et al. (2024). Formalizing and Benchmarking Prompt Injection Attacks and Defenses. *USENIX Security '24*. arXiv:2310.12815",
        f"- Perez, F. & Ribeiro, I. (2022). Ignore Previous Prompt. arXiv:2211.09527",
        f"- Greshake, K. et al. (2023). Not what you've signed up for. arXiv:2302.12173",
        f"",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse benchmark results")
    parser.add_argument("summary", help="Path to summary_*.json file")
    parser.add_argument("--markdown", action="store_true", help="Also write a Markdown report")
    parser.add_argument("--csv-only", action="store_true", help="Write CSVs only, skip console output")
    args = parser.parse_args()

    path = Path(args.summary)
    if not path.exists():
        print(f"Error: file not found: {path}")
        sys.exit(1)

    summary    = load_summary(args.summary)
    output_dir = path.parent

    analyse(summary, output_dir, write_md=args.markdown)
