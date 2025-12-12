"""
summarize_pos_runs.py
combine key metrics across diff run_pos_pipeline.py outputs
for each run dir (default: all subfolders of ./runs), script collects
- style baselines: mu_you_on_you, mu_gpt_on_gpt
- rewritten vs original avg rel. style scores (parsed from log.txt)
- # POS changes (len of models/pos_change_log.json)
outputs simple table to stdout
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple


STYLE_REL_RE = re.compile(
    r"Average normalized relative style \(you - GPT\):\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)


def parse_style_scores(log_path: Path) -> Tuple[Optional[float], Optional[float]]:
    """
    parse avg rel style scores for rewritten, orig essays from log.txt
    return (rewritten_style_rel, original_style_rel)
    """
    try:
        text = log_path.read_text(encoding="utf-8")
    except Exception:
        return None, None

    #sections: YOUR ESSAY, then ORIGINAL GPT ESSAY
    parts = text.split("=== YOUR ESSAY ===")
    if len(parts) < 2:
        return None, None
    your_section = parts[1]
    orig_split = your_section.split("=== ORIGINAL GPT ESSAY ===")
    orig_section = orig_split[1] if len(orig_split) > 1 else ""

    def find_first(section: str) -> Optional[float]:
        m = STYLE_REL_RE.search(section)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
        return None

    rewritten = find_first(your_section)
    original = find_first(orig_section)
    return rewritten, original


def summarize_run(run_dir: Path) -> Optional[Dict[str, object]]:
    baselines_path = run_dir / "models" / "style_baselines.json"
    pos_change_log = run_dir / "models" / "pos_change_log.json"
    log_path = run_dir / "log.txt"

    if not baselines_path.exists():
        return None

    try:
        baselines = json.loads(baselines_path.read_text(encoding="utf-8"))
    except Exception:
        baselines = {}

    try:
        pos_changes = json.loads(pos_change_log.read_text(encoding="utf-8"))
        num_changes = len(pos_changes) if isinstance(pos_changes, list) else None
    except Exception:
        num_changes = None

    rewritten_style, original_style = parse_style_scores(log_path)

    return {
        "run": run_dir.name,
        "mu_you_on_you": baselines.get("mu_you_on_you"),
        "mu_gpt_on_gpt": baselines.get("mu_gpt_on_gpt"),
        "rewritten_style_rel": rewritten_style,
        "original_style_rel": original_style,
        "num_pos_changes": num_changes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Summarize POS pipeline metrics across run dirs"
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs",
        help="Parent dir containing run folders",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="path to write TSV of per-run metrics",
    )
    args = parser.parse_args()

    runs_root = Path(args.runs_dir).resolve()
    if not runs_root.exists():
        raise FileNotFoundError(runs_root)

    rows = []
    for sub in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        rec = summarize_run(sub)
        if rec:
            rows.append(rec)

    if not rows:
        print("No runs found with models/style_baselines.json under", runs_root)
        return

    #get per-run improvement and comb. average
    for r in rows:
        rew = r.get("rewritten_style_rel")
        orig = r.get("original_style_rel")
        if rew is not None and orig is not None:
            r["style_delta"] = rew - orig
        else:
            r["style_delta"] = None

    headers = [
        "run",
        "mu_you_on_you",
        "mu_gpt_on_gpt",
        "rewritten_style_rel",
        "original_style_rel",
        "style_delta",
        "num_pos_changes",
    ]
    lines = []
    lines.append("\t".join(headers))
    for r in rows:
        lines.append(
            "\t".join(
                str(r.get(h, "")) if r.get(h, "") is not None else "" for h in headers
            )
        )
    print("\n".join(lines))

    #overall avgs.
    def avg(field: str) -> Optional[float]:
        vals = [r[field] for r in rows if r.get(field) is not None]
        if not vals:
            return None
        return sum(vals) / len(vals)

    averages_lines = [
        "# Averages across runs",
        f"avg_mu_you_on_you\t{avg('mu_you_on_you')}",
        f"avg_mu_gpt_on_gpt\t{avg('mu_gpt_on_gpt')}",
        f"avg_rewritten_style_rel\t{avg('rewritten_style_rel')}",
        f"avg_original_style_rel\t{avg('original_style_rel')}",
        f"avg_style_delta\t{avg('style_delta')}",
    ]
    print("\n".join(averages_lines))

    if args.out:
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_lines = lines + [""] + averages_lines
        out_path.write_text("\n".join(out_lines), encoding="utf-8")
        print(f"\n[SAVED] Summary written to {out_path}")


if __name__ == "__main__":
    main()
