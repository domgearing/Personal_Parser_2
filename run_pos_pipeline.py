"""
run_pos_experiment.py
Main pipeline for POS style scoring and rewriting
Does the following:
- preprocess personal + GPT texts
- build POS trees + POS-PCFGs
- compute style baselines
- build style stats (POS n-grams etc.)
- rewrite GPT essays toward personal style
- analyze grammar + POS changes
- generate grammar fingerprint vis
- organize everything into runs/YYYY-MM-DD_HH-MM-SS_LABEL/
- build simple HTML dash for each run
- opt batch over multiple GPT essays

Run CMD ex:

Single pair (one personal, one GPT essay):

    python run_pos_pipeline.py \
        --label alcohol_dbb \
        --personal-text personal_data/personal_dbb.txt \
        --gpt-text gpt_data/gpt_dbb.txt \
        --gpt-cands data/gpt_candidates.jsonl

batch over all GPT *.txt files in folder (same personal corpus):

    python run_pos_pipeline.py \
        --label batch_all \
        --personal-dir personal_data \
        --gpt-dir gpt_data \ 
        --gpt-cands data/gpt_candidates.jsonl

turn individual stages on/off with --no-XXX flags
see argparse section at bottom
"""

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

from nltk.tokenize import sent_tokenize

#path to repo (where all scripts are)
SCRIPT_DIR = Path(__file__).resolve().parent

#helper funcs

def run_cmd(
    cmd: List[str],
    log_file: Path,
    step_name: str,
    debug: bool = False,
    cwd: Optional[Path] = None,
):
    """
    run shell command, direct output to console and log file
    raises if command fails
    """
    start = time.perf_counter()
    print(f"\n[STEP] {step_name}")
    print("  $", " ".join(cmd))
    if cwd:
        print(f"  (cwd={cwd})")

    with log_file.open("a", encoding="utf-8") as lf:
        lf.write(f"\n\n===== {step_name} =====\n")
        lf.write("CMD: " + " ".join(cmd) + "\n")
        if cwd:
            lf.write(f"cwd: {cwd}\n")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(cwd) if cwd else None,
        )
        for line in proc.stdout:
            line = line.rstrip("\n")
            lf.write(line + "\n")
            if debug:
                print("   ", line)

        proc.wait()
        elapsed = time.perf_counter() - start
        lf.write(f"\n[STEP COMPLETED] {step_name} in {elapsed:.2f}s\n")

    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed with code {proc.returncode}: {' '.join(cmd)}"
        )
    print(f"[OK] {step_name} (t={elapsed:.2f}s)")


def first_sentence_from_file(path: Path) -> str:
    """
    return first sent from txt file (sent tokenized) fallback to ''
    avoids taking multi-sent lines by splitting with sent_tokenize
    """
    try:
        text = path.read_text(encoding="utf-8")
        for s in sent_tokenize(text):
            stripped = s.strip()
            if stripped:
                return stripped
    except Exception:
        pass
    return ""


def first_n_sentences(path: Path, n: int = 2) -> List[str]:
    """
    return first n non empty sents from txt file
    """
    out: List[str] = []
    try:
        text = path.read_text(encoding="utf-8")
        for s in sent_tokenize(text):
            stripped = s.strip()
            if stripped:
                out.append(stripped)
            if len(out) >= n:
                break
    except Exception:
        pass
    return out


def timestamp_label(label: Optional[str]) -> str:
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if label:
        return f"{ts}_{label}"
    return ts


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

#html dashboard for viz and results

def build_dashboard_html(run_dir: Path, personal_text: Path, gpt_texts: List[Path]):
    """
    make HTML page in run_dir that links all gen. figs
    summarizes JSON outputs
    """
    html_path = run_dir / "dashboard.html"

    figs = sorted(
        [p for p in (run_dir / "viz").glob("*.png")] if (run_dir / "viz").exists() else []
    )

    #JSON summaries
    style_baselines = run_dir / "models" / "style_baselines.json"
    style_stats = run_dir / "models" / "style_stats_pos.json"
    pos_change_log = run_dir / "models" / "pos_change_log.json"

    def load_json(path: Path):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def pretty_json(obj, max_chars: int = 1800) -> str:
        if obj is None:
            return "(missing)"
        txt = json.dumps(obj, indent=2)
        return txt if len(txt) <= max_chars else txt[:max_chars] + "... (truncated)"

    baselines_data = load_json(style_baselines)
    stats_data = load_json(style_stats)
    pos_changes = load_json(pos_change_log)

    with html_path.open("w", encoding="utf-8") as f:
        f.write("<html><head><meta charset='utf-8'>\n")
        f.write("<title>Grammar Fingerprint Dashboard</title>\n")
        f.write(
            "<style>"
            "body{font-family:system-ui, sans-serif;margin:30px;line-height:1.5;}"
            "section{margin-bottom:28px;}"
            "h1,h2,h3{margin-bottom:10px;}"
            "ul{padding-left:20px;}"
            "img{max-width:100%;margin:10px 0;border:1px solid #ccc;}"
            "pre{background:#f5f5f5;padding:10px;overflow-x:auto;white-space:pre-wrap;}"
            "table{border-collapse:collapse;}"
            "td,th{border:1px solid #ddd;padding:6px 10px;}"
            ".pill{display:inline-block;padding:6px 10px;border-radius:6px;background:#f0f4ff;}"
            ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;}"
            "</style>\n"
        )
        f.write("</head><body>\n")
        f.write(f"<h1>Grammar Fingerprint Run: {run_dir.name}</h1>\n")

        #inputs
        f.write("<section><h2>Inputs</h2>\n<ul>\n")
        f.write(f"<li>Personal text: {personal_text}</li>\n")
        for g in gpt_texts:
            f.write(f"<li>GPT text: {g}</li>\n")
        f.write("</ul></section>\n")

        #baselines
        if baselines_data:
            f.write("<section><h2>Style Baselines</h2>\n")
            f.write("<div class='grid'>\n")
            f.write(
                f"<div class='pill'>mu_you_on_you: {baselines_data.get('mu_you_on_you')}</div>\n"
            )
            f.write(
                f"<div class='pill'>mu_gpt_on_gpt: {baselines_data.get('mu_gpt_on_gpt')}</div>\n"
            )
            f.write("</div>\n")
            f.write(
                "<details><summary>View raw style_baselines.json</summary>"
                f"<pre>{pretty_json(baselines_data)}</pre></details>\n"
            )
            f.write("</section>\n")

        #POS style stats summary
        if stats_data:
            f.write("<section><h2>POS Style Stats</h2>\n")

            punctuation_tags = {
                ",",
                ".",
                ":",
                "``",
                "''",
                "(",
                ")",
                "-LRB-",
                "-RRB-",
                "--",
                ";",
                "!",
                "?",
            }

            def top_pos_counts(d: dict, k: str = "pos_unigrams", n: int = 8):
                counts = {
                    tag: cnt
                    for tag, cnt in d.get(k, {}).items()
                    if tag not in punctuation_tags
                }
                total = sum(counts.values()) or 1
                top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:n]
                return [(tag, cnt, round(100 * cnt / total, 1)) for tag, cnt in top]

            you_stats = stats_data.get("you", {})
            gpt_stats = stats_data.get("gpt", {})
            f.write("<div class='grid'>\n")
            f.write("<div><h3>Top POS (You)</h3><table>")
            f.write("<tr><th>Tag</th><th>Count</th><th>%</th></tr>")
            for tag, cnt, pct in top_pos_counts(you_stats):
                f.write(f"<tr><td>{tag}</td><td>{cnt}</td><td>{pct}%</td></tr>")
            f.write("</table></div>\n")
            f.write("<div><h3>Top POS (GPT)</h3><table>")
            f.write("<tr><th>Tag</th><th>Count</th><th>%</th></tr>")
            for tag, cnt, pct in top_pos_counts(gpt_stats):
                f.write(f"<tr><td>{tag}</td><td>{cnt}</td><td>{pct}%</td></tr>")
            f.write("</table></div>\n")
            f.write("</div>\n")
            f.write(
                "<details><summary>View raw style_stats_pos.json</summary>"
                f"<pre>{pretty_json(stats_data)}</pre></details>\n"
            )
            f.write("</section>\n")

        #POS change summary
        if isinstance(pos_changes, list) and pos_changes:
            f.write("<section><h2>POS Change Log</h2>\n")
            f.write(f"<p>Total rewritten sentences: {len(pos_changes)}</p>\n")
            f.write("<ol>\n")
            for rec in pos_changes[:5]:
                orig = rec.get("orig", "")
                new = rec.get("new", "")
                f.write(
                    "<li><strong>Sentence "
                    f"{rec.get('sent_idx', '?')}:</strong><br>"
                    f"<em>Orig:</em> {orig}<br>"
                    f"<em>New:</em> {new}</li>\n"
                )
            if len(pos_changes) > 5:
                f.write(f"<li>…and {len(pos_changes) - 5} more</li>\n")
            f.write("</ol>\n")
            f.write(
                "<details><summary>View raw pos_change_log.json</summary>"
                f"<pre>{pretty_json(pos_changes)}</pre></details>\n"
            )
            f.write("</section>\n")

        if figs:
            f.write("<section><h2>Visualizations</h2>\n")
            categories = {
                "Barplots": [],
                "Heatmaps": [],
                "Graphs": [],
                "Scatter": [],
                "Fingerprints": [],
                "Other": [],
            }

            for p in figs:
                name = p.name.lower()
                if "heatmap" in name:
                    categories["Heatmaps"].append(p)
                elif "graph" in name:
                    categories["Graphs"].append(p)
                elif "scatter" in name:
                    categories["Scatter"].append(p)
                elif "radial" in name or "fingerprint" in name:
                    categories["Fingerprints"].append(p)
                elif "bar" in name or "bigrams" in name or "pos.png" in name:
                    categories["Barplots"].append(p)
                else:
                    categories["Other"].append(p)

            for title, group in categories.items():
                if not group:
                    continue
                f.write(f"<h3>{title}</h3>\n<div class='grid'>\n")
                for p in group:
                    rel = p.relative_to(run_dir)
                    f.write(
                        f"<div><div style='font-weight:600'>{rel.name}</div>"
                        f"<img src='{rel.as_posix()}' /></div>\n"
                    )
                f.write("</div>\n")
            f.write("</section>\n")

        f.write("</body></html>\n")

    print(f"[SAVED] Dashboard HTML → {html_path}")

#Pipeline runner for 1 GPT essay


def run_single_experiment(
    run_dir: Path,
    personal_text: Path,
    gpt_text: Path,
    gpt_cands: Optional[Path],
    debug: bool = False,
    run_preprocess: bool = True,
    run_trees: bool = True,
    run_pcfg: bool = True,
    run_pcfg_summary: bool = True,
    run_cky_debug: bool = True,
    run_baselines: bool = True,
    run_style_stats: bool = True,
    run_rewrite: bool = True,
    run_diagnose: bool = True,
    run_grammar_viz: bool = True,
    run_pos_change_viz: bool = True,
    run_compare_styles: bool = True,
    run_graphviz_pcfg: bool = True,
    run_tree_viz: bool = True,
    run_cky_viz: bool = True,
):
    """
    Execute POS pipeline for one (personal, GPT) pair
    All outputs written inside run_dir:
       run_dir/models
       run_dir/personal_data
       run_dir/gpt_data
       run_dir/viz
       run_dir/log.txt
    """
    run_dir = run_dir.resolve()
    ensure_dir(run_dir)
    ensure_dir(run_dir / "models")
    ensure_dir(run_dir / "viz")
    ensure_dir(run_dir / "personal_data")
    ensure_dir(run_dir / "gpt_data")

    log_file = run_dir / "log.txt"

    #copy raw inputs into run dir for record keeping
    personal_raw = run_dir / "personal_data" / personal_text.name
    gpt_raw = run_dir / "gpt_data" / gpt_text.name

    shutil.copy2(personal_text, personal_raw)
    shutil.copy2(gpt_text, gpt_raw)

    if gpt_cands is not None and gpt_cands.exists():
        ensure_dir(run_dir / "data")
        shutil.copy2(gpt_cands, run_dir / "data" / gpt_cands.name)

    #paths inside run (personal/gpt cleaned text + trees + models)
    personal_clean_jsonl = (
        run_dir / "personal_data" / f"{personal_text.stem}_clean.jsonl"
    )
    gpt_clean_jsonl = run_dir / "gpt_data" / f"{gpt_text.stem}_clean.jsonl"

    personal_clean_txt = run_dir / "personal_data" / f"{personal_text.stem}_clean.txt"
    gpt_clean_txt = run_dir / "gpt_data" / f"{gpt_text.stem}_clean.txt"

    personal_trees = run_dir / "personal_data" / f"{personal_text.stem}_pos_trees.jsonl"
    gpt_trees = run_dir / "gpt_data" / f"{gpt_text.stem}_pos_trees.jsonl"

    pcfg_personal = run_dir / "models" / "pcfg_personal_pos.json"
    pcfg_gpt = run_dir / "models" / "pcfg_gpt_pos.json"
    style_stats_out = run_dir / "models" / "style_stats_pos.json"
    baselines_out = run_dir / "models" / "style_baselines.json"
    rewrite_out = run_dir / "rewritten_gpt_essay_relative_pos.txt"
    pos_change_log = run_dir / "models" / "pos_change_log.json"

    #preprocess: process_sample_text.py
    if run_preprocess:
        run_cmd(
            [
                sys.executable,
                str(SCRIPT_DIR / "process_sample_text.py"),
                "--in-dir",
                str(run_dir / "personal_data"),
                "--out-path",
                str(personal_clean_jsonl),
            ],
            log_file,
            "Preprocess personal text",
            debug,
            cwd=SCRIPT_DIR,
        )

        run_cmd(
            [
                sys.executable,
                str(SCRIPT_DIR / "process_sample_text.py"),
                "--in-dir",
                str(run_dir / "gpt_data"),
                "--out-path",
                str(gpt_clean_jsonl),
            ],
            log_file,
            "Preprocess GPT text",
            debug,
            cwd=SCRIPT_DIR,
        )
    else:
        print("[SKIP] Preprocessing – using existing *_clean.txt")
        #assume user already made 
        if (
            not personal_clean_txt.exists()
            or not gpt_clean_txt.exists()
            or not personal_clean_jsonl.exists()
            or not gpt_clean_jsonl.exists()
        ):
            raise FileNotFoundError(
                "Expected pre-cleaned files but they do not exist: "
                f"{personal_clean_txt}, {gpt_clean_txt}, "
                f"{personal_clean_jsonl}, {gpt_clean_jsonl}"
            )
            
    #POS trees -  make_POS_trees.py
    if run_trees:
        run_cmd(
            [
                sys.executable,
                str(SCRIPT_DIR / "make_POS_trees.py"),
                "--in-path",
                str(personal_clean_jsonl),
                "--out-path",
                str(personal_trees),
            ],
            log_file,
            "Build POS trees for personal text",
            debug,
            cwd=SCRIPT_DIR,
        )
        run_cmd(
            [
                sys.executable,
                str(SCRIPT_DIR / "make_POS_trees.py"),
                "--in-path",
                str(gpt_clean_jsonl),
                "--out-path",
                str(gpt_trees),
            ],
            log_file,
            "Build POS trees for GPT text",
            debug,
            cwd=SCRIPT_DIR,
        )
    else:
        print("[SKIP] POS tree building")

    #viz few trees as Graphviz PDFs
    if run_tree_viz and run_trees:
        tree_viz_dir_personal = run_dir / "viz" / "trees_personal"
        tree_viz_dir_gpt = run_dir / "viz" / "trees_gpt"
        run_cmd(
            [
                sys.executable,
                str(SCRIPT_DIR / "visualize_trees_graphviz.py"),
                "--jsonl",
                str(personal_trees),
                "--out-dir",
                str(tree_viz_dir_personal),
                "--num-examples",
                "3",
            ],
            log_file,
            "Visualize POS trees (personal)",
            debug,
            cwd=SCRIPT_DIR,
        )
        run_cmd(
            [
                sys.executable,
                str(SCRIPT_DIR / "visualize_trees_graphviz.py"),
                "--jsonl",
                str(gpt_trees),
                "--out-dir",
                str(tree_viz_dir_gpt),
                "--num-examples",
                "3",
            ],
            log_file,
            "Visualize POS trees (GPT)",
            debug,
            cwd=SCRIPT_DIR,
        )

    #POS-PCFG: pcfg_POS.py
    
    if run_pcfg:
        run_cmd(
            [
                sys.executable,
                str(SCRIPT_DIR / "pcfg_POS.py"),
                "--trees",
                str(personal_trees),
                "--out",
                str(pcfg_personal),
            ],
            log_file,
            "Estimate POS-based PCFG for personal text",
            debug,
            cwd=SCRIPT_DIR,
        )
        run_cmd(
            [
                sys.executable,
                str(SCRIPT_DIR / "pcfg_POS.py"),
                "--trees",
                str(gpt_trees),
                "--out",
                str(pcfg_gpt),
            ],
            log_file,
            "Estimate POS-based PCFG for GPT text",
            debug,
            cwd=SCRIPT_DIR,
        )
    else:
        print("[SKIP] POS-PCFG estimation")

    #PCFG summaries/comparisons/graph fragments
    if run_pcfg_summary:
        run_cmd(
            [
                sys.executable,
                str(SCRIPT_DIR / "describe_pcfg.py"),
                "--grammar",
                str(pcfg_personal),
                "--csv-out",
                str(run_dir / "models" / "nonterminals_personal.csv"),
                "--md-out",
                str(run_dir / "models" / "nonterminals_personal.md"),
            ],
            log_file,
            "Describe PCFG (personal)",
            debug,
            cwd=SCRIPT_DIR,
        )
        run_cmd(
            [
                sys.executable,
                str(SCRIPT_DIR / "describe_pcfg.py"),
                "--grammar",
                str(pcfg_gpt),
                "--csv-out",
                str(run_dir / "models" / "nonterminals_gpt.csv"),
                "--md-out",
                str(run_dir / "models" / "nonterminals_gpt.md"),
            ],
            log_file,
            "Describe PCFG (GPT)",
            debug,
            cwd=SCRIPT_DIR,
        )
        run_cmd(
            [
                sys.executable,
                str(SCRIPT_DIR / "analyze_grammar.py"),
                "--grammar",
                str(pcfg_personal),
                "--top-k",
                "6",
            ],
            log_file,
            "Analyze grammar entropy/top rules (personal)",
            debug,
            cwd=SCRIPT_DIR,
        )
        run_cmd(
            [
                sys.executable,
                str(SCRIPT_DIR / "analyze_grammar.py"),
                "--grammar",
                str(pcfg_gpt),
                "--top-k",
                "6",
            ],
            log_file,
            "Analyze grammar entropy/top rules (GPT)",
            debug,
            cwd=SCRIPT_DIR,
        )

    if run_compare_styles:
        run_cmd(
            [
                sys.executable,
                str(SCRIPT_DIR / "compare_styles.py"),
                "--personal-grammar",
                str(pcfg_personal),
                "--gpt-grammar",
                str(pcfg_gpt),
                "--top-k",
                "12",
            ],
            log_file,
            "Compare structural style gaps (personal vs GPT)",
            debug,
            cwd=SCRIPT_DIR,
        )

    if run_graphviz_pcfg:
        graph_out_personal = run_dir / "viz" / "pcfg_fragment_personal"
        graph_out_gpt = run_dir / "viz" / "pcfg_fragment_gpt"
        run_cmd(
            [
                sys.executable,
                str(SCRIPT_DIR / "visualize_graph_pcfg.py"),
                "--grammar",
                str(pcfg_personal),
                "--focus-lhs",
                "S,NP,VP,PP,SBAR",
                "--top-n-rules",
                "5",
                "--out",
                str(graph_out_personal),
            ],
            log_file,
            "Graphviz PCFG fragment (personal)",
            debug,
            cwd=SCRIPT_DIR,
        )
        run_cmd(
            [
                sys.executable,
                str(SCRIPT_DIR / "visualize_graph_pcfg.py"),
                "--grammar",
                str(pcfg_gpt),
                "--focus-lhs",
                "S,NP,VP,PP,SBAR",
                "--top-n-rules",
                "5",
                "--out",
                str(graph_out_gpt),
            ],
            log_file,
            "Graphviz PCFG fragment (GPT)",
            debug,
            cwd=SCRIPT_DIR,
        )

    #CKY sanity check with learned personal grammar
    if run_cky_debug:
        if not pcfg_personal.exists():
            raise FileNotFoundError(
                f"Missing personal PCFG at {pcfg_personal}; cannot run CKY debug."
            )
        run_cmd(
            [
                sys.executable,
                str(SCRIPT_DIR / "cky_POS.py"),
                "--grammar",
                str(pcfg_personal),
                "--text",
                str(gpt_clean_txt),
                "--sent-idx",
                "0",
            ],
            log_file,
            "CKY sanity check on GPT text (personal grammar)",
            debug,
            cwd=SCRIPT_DIR,
        )
    else:
        print("[SKIP] CKY sanity check")

    #CKY chart vis (two sents each)
    if run_cky_viz:
        personal_sents = first_n_sentences(personal_clean_txt, n=2)
        gpt_sents = first_n_sentences(gpt_clean_txt, n=2)

        if personal_sents and pcfg_personal.exists():
            for idx, s in enumerate(personal_sents):
                run_cmd(
                    [
                        sys.executable,
                        str(SCRIPT_DIR / "visualize_cky_chart.py"),
                        "--grammar",
                        str(pcfg_personal),
                        "--sent",
                        s,
                        "--pos-grammar",
                        "--out",
                        str(run_dir / "viz" / f"cky_chart_personal_{idx}.png"),
                    ],
                    log_file,
                    f"Visualize CKY chart on personal sentence {idx}",
                    debug,
                    cwd=SCRIPT_DIR,
                )
        else:
            print("[SKIP] CKY chart viz (missing personal sentences or PCFG)")

        if gpt_sents and pcfg_gpt.exists():
            for idx, s in enumerate(gpt_sents):
                run_cmd(
                    [
                        sys.executable,
                        str(SCRIPT_DIR / "visualize_cky_chart.py"),
                        "--grammar",
                        str(pcfg_gpt),
                        "--sent",
                        s,
                        "--pos-grammar",
                        "--out",
                        str(run_dir / "viz" / f"cky_chart_gpt_{idx}.png"),
                    ],
                    log_file,
                    f"Visualize CKY chart on GPT sentence {idx}",
                    debug,
                    cwd=SCRIPT_DIR,
                )
        else:
            print("[SKIP] CKY chart viz (missing GPT sentences or PCFG)")

    #style stats over POS (n-grams etc)
    if run_style_stats:
        #build_style_stats_pos.py reads trees/PCFGs and writes
        #models/style_stats_pos.json
        run_cmd(
            [
                sys.executable,
                str(SCRIPT_DIR / "build_style_stats_pos.py"),
                "--you-text",
                str(personal_clean_txt),
                "--gpt-text",
                str(gpt_clean_txt),
                "--out",
                str(style_stats_out),
            ],
            log_file,
            "Build POS style statistics (Pw, Ppos, etc.)",
            debug,
            cwd=SCRIPT_DIR,
        )
    else:
        print("[SKIP] style_stats_pos")

    #CKY scoring + baselines
    # CKY_pos mainly for debug per-sent parses 
    if run_baselines:
        #compute_baselines.py write models/style_baselines.json
        if not pcfg_personal.exists() or not pcfg_gpt.exists():
            raise FileNotFoundError(
                "Missing PCFGs for baselines. Expected:\n"
                f"  personal: {pcfg_personal}\n"
                f"  gpt:      {pcfg_gpt}"
            )
        run_cmd(
            [
                sys.executable,
                str(SCRIPT_DIR / "compute_baselines.py"),
                "--mine-train",
                str(personal_clean_txt),
                "--gpt-train",
                str(gpt_clean_txt),
                "--personal_grammar",
                str(pcfg_personal),
                "--gpt_grammar",
                str(pcfg_gpt),
                "--out-baselines",
                str(baselines_out),
            ],
            log_file,
            "Compute style baselines (mu_you_on_you, mu_gpt_on_gpt)",
            debug,
            cwd=SCRIPT_DIR,
        )
    else:
        print("[SKIP] baselines (style_baselines.json)")

    #rewrite GPT essay toward your style (POS)
    if run_rewrite:
        if not baselines_out.exists():
            raise FileNotFoundError(
                f"Missing style baselines at {baselines_out}; run baselines first or point --baselines to an existing JSON"
            )
        if not pcfg_personal.exists() or not pcfg_gpt.exists():
            raise FileNotFoundError(
                "Missing PCFGs for rewrite stage. Expected:\n"
                f"  personal: {pcfg_personal}\n"
                f"  gpt:      {pcfg_gpt}"
            )
        rewrite_cmd = [
            sys.executable,
            str(SCRIPT_DIR / "rewrite_POS.py"),
            "--mine",
            str(personal_clean_txt),
            "--gpt",
            str(gpt_clean_txt),
            "--you-grammar",
            str(pcfg_personal),
            "--gpt-grammar",
            str(pcfg_gpt),
            "--baselines",
            str(baselines_out),
            "--out-essay",
            str(rewrite_out),
            "--out-log",
            str(pos_change_log),
        ]
        if gpt_cands is not None and gpt_cands.exists():
            rewrite_cmd += ["--cands", str(run_dir / "data" / gpt_cands.name)]

        run_cmd(
            rewrite_cmd,
            log_file,
            "Rewrite GPT essay toward personal style (rewrite_POS.py)",
            debug,
            cwd=SCRIPT_DIR,
        )
    else:
        print("[SKIP] rewrite_POS")

    #diagnose sent style (per-sent plots + tables)
    if run_diagnose:
        if not style_stats_out.exists():
            raise FileNotFoundError(
                f"Missing style stats at {style_stats_out}; run style-stats stage first"
            )
        if not pcfg_personal.exists():
            raise FileNotFoundError(
                f"Missing personal PCFG for diagnosis: {pcfg_personal}"
            )
        #python diagnose_sentence_style_pos.py --gpt <clean> --mine <clean>
        run_cmd(
            [
                sys.executable,
                str(SCRIPT_DIR / "diagnose_sentence_style_pos.py"),
                "--text",
                str(gpt_clean_txt),
                "--sent-idx",
                "0",
                "--style-stats",
                str(style_stats_out),
                "--grammar",
                str(pcfg_personal),
            ],
            log_file,
            "Diagnose per-sentence style (diagnose_sentence_style_pos.py)",
            debug,
            cwd=SCRIPT_DIR,
        )
        #if script writes figures, move under run_dir/viz
        for p in Path(run_dir).glob("sent_style_*.png"):
            shutil.move(str(p), run_dir / "viz" / p.name)
    else:
        print("[SKIP] diagnose_sentence_style_pos")

    #grammar fingerprint + POS vis
    if run_grammar_viz:
        if not pos_change_log.exists():
            raise FileNotFoundError(
                f"Missing POS change log at {pos_change_log}; run rewrite step first."
            )
        run_cmd(
            [sys.executable, str(SCRIPT_DIR / "grammar_fingerprint_viz.py")],
            log_file,
            "Grammar fingerprint visualizations (PCFG + POS bigrams)",
            debug,
            cwd=run_dir,
        )
    else:
        print("[SKIP] grammar_fingerprint_viz")

    #POS change agg./Sankey etc.
    if run_pos_change_viz:
        if not pos_change_log.exists():
            raise FileNotFoundError(
                f"Missing POS change log at {pos_change_log}; run rewrite step first"
            )
        run_cmd(
            [sys.executable, str(SCRIPT_DIR / "analyze_pos_changes.py")],
            log_file,
            "Analyze POS changes (aggregate + visualizations)",
            debug,
            cwd=run_dir,
        )
        pos_change_plots = run_dir / "models" / "pos_change_plots"
        if pos_change_plots.exists():
            for p in pos_change_plots.glob("*.png"):
                shutil.copy2(p, run_dir / "viz" / p.name)
            for p in pos_change_plots.glob("*.html"):
                shutil.copy2(p, run_dir / "viz" / p.name)
    else:
        print("[SKIP] analyze_pos_changes")

    print(f"\n[RUN COMPLETE] Outputs collected in {run_dir}")

#main
def main():
    parser = argparse.ArgumentParser(
        description="main for POS-based style pipeline (personal vs GPT)"
    )
    parser.add_argument("--label", type=str, default=None, help="Label for this run.")
    parser.add_argument(
        "--personal-text",
        type=str,
        required=True,
        help="Path to personal txt file",
    )
    parser.add_argument(
        "--personal-dir",
        type=str,
        default=None,
        help="If batching with --gpt-dir, dir containing personal_*.txt files to match by name",
    )

    #single GPT essay or dir of GPT essays
    parser.add_argument(
        "--gpt-text",
        type=str,
        default=None,
        help="Single GPT essay txt file",
    )
    parser.add_argument(
        "--gpt-dir",
        type=str,
        default=None,
        help="run separate experiment for every *.txt in dir",
    )

    parser.add_argument(
        "--gpt-cands",
        type=str,
        default=None,
        help="JSONL file with GPT paraphrase cands",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="direct all subprocess output to console",
    )

    #turn stages on/off
    parser.add_argument("--no-preprocess", action="store_true")
    parser.add_argument("--no-trees", action="store_true")
    parser.add_argument("--no-pcfg", action="store_true")
    parser.add_argument("--no-pcfg-summary", action="store_true")
    parser.add_argument("--no-cky", action="store_true")
    parser.add_argument("--no-baselines", action="store_true")
    parser.add_argument("--no-style-stats", action="store_true")
    parser.add_argument("--no-rewrite", action="store_true")
    parser.add_argument("--no-diagnose", action="store_true")
    parser.add_argument("--no-grammar-viz", action="store_true")
    parser.add_argument("--no-pos-change-viz", action="store_true")
    parser.add_argument("--no-compare-styles", action="store_true")
    parser.add_argument("--no-graphviz-pcfg", action="store_true")
    parser.add_argument("--no-tree-viz", action="store_true")
    parser.add_argument("--no-cky-viz", action="store_true")

    args = parser.parse_args()

    personal_text = Path(args.personal_text).resolve()
    if not personal_text.exists():
        raise FileNotFoundError(personal_text)
    personal_dir = (
        Path(args.personal_dir).resolve()
        if args.personal_dir
        else personal_text.parent
    )
    if not personal_dir.exists():
        raise FileNotFoundError(f"Personal dir not found: {personal_dir}")

    if (args.gpt_text is None) == (args.gpt_dir is None):
        raise ValueError(
            "specify exactly one of --gpt-text or --gpt-dir (not both)"
        )

    gpt_cands = Path(args.gpt_cands).resolve() if args.gpt_cands else None

    base_run_root = Path("runs").resolve()
    ensure_dir(base_run_root)

    #collect all GPT essay files to run
    if args.gpt_text:
        gpt_files = [Path(args.gpt_text).resolve()]
    else:
        gpt_root = Path(args.gpt_dir).resolve()
        gpt_files = sorted(gpt_root.glob("*.txt"))
        if not gpt_files:
            raise FileNotFoundError(f"No *.txt files found in {gpt_root}")

    all_run_dirs = []

    for gpt_file in gpt_files:
        run_name = timestamp_label(args.label or gpt_file.stem)
        run_dir = base_run_root / run_name
        all_run_dirs.append(run_dir)

        #in batch mode, match personal_*.txt to gpt_*.txt by name prefix
        if args.gpt_dir:
            personal_candidate = personal_dir / gpt_file.name.replace("gpt_", "personal_", 1)
            if not personal_candidate.exists():
                raise FileNotFoundError(
                    f"Expected matching personal file for {gpt_file.name}: {personal_candidate}"
                )
            personal_input = personal_candidate
        else:
            personal_input = personal_text

        #pick cand file matching topic 
        topic_stem = gpt_file.stem.replace("gpt_", "", 1)
        cand_path = None
        #prefer data/gpt_candidates_<topic>.jsonl then runs/<...>/data
        candidate_names = [
            f"data/gpt_candidates_{topic_stem}.jsonl",
            f"data/gpt_candidates.jsonl" if topic_stem == "dbb" else None,
        ]
        for name in candidate_names:
            if not name:
                continue
            p = Path(name)
            if p.exists():
                cand_path = p.resolve()
                break

        run_single_experiment(
            run_dir=run_dir,
            personal_text=personal_input,
            gpt_text=gpt_file,
            gpt_cands=cand_path,
            debug=args.debug,
            run_preprocess=not args.no_preprocess,
            run_trees=not args.no_trees,
            run_pcfg=not args.no_pcfg,
            run_pcfg_summary=not args.no_pcfg_summary,
            run_cky_debug=not args.no_cky,
            run_baselines=not args.no_baselines,
            run_style_stats=not args.no_style_stats,
            run_rewrite=not args.no_rewrite,
            run_diagnose=not args.no_diagnose,
            run_grammar_viz=not args.no_grammar_viz,
            run_pos_change_viz=not args.no_pos_change_viz,
            run_compare_styles=not args.no_compare_styles,
            run_graphviz_pcfg=not args.no_graphviz_pcfg,
            run_tree_viz=not args.no_tree_viz,
            run_cky_viz=not args.no_cky_viz,
        )

    #build dash per run
    for run_dir, gpt_file in zip(all_run_dirs, gpt_files):
        if args.gpt_dir:
            personal_input = personal_dir / gpt_file.name.replace("gpt_", "personal_", 1)
        else:
            personal_input = personal_text
        build_dashboard_html(
            run_dir=run_dir,
            personal_text=personal_input,
            gpt_texts=[gpt_file],
        )

    print("\n[ALL DONE] Experiments finished. See 'runs/' for outputs + dashboards")


if __name__ == "__main__":
    main()
