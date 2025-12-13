# Personal_Parser_2

Final project for Language and Computation at Yale University. Builds POS-based PCFG grammars from a personal corpus and from GPT outputs, scores sentences and essays for how close they are to the personal vs GPT grammar, and rewrites GPT essays toward the user's personal style using POS-level rewrites and optional GPT paraphrase candidates.

---

## Overview

This project compares structural (POS/PCFG) patterns between a user's writing and GPT-generated text, then adapts GPT output toward the user's style. Core steps implemented across the repository:

- Clean and preprocess personal and GPT text.
- Build POS trees and estimate POS-based PCFGs (CNF productions + probabilities).
- Compute style baselines and POS/lexical statistics.
- Score sentences/essays for "you-ness" vs "GPT-ness" using CKY-style parse scores.
- Rewrite GPT essays sentence-by-sentence using POS-level rewrites and optional GPT paraphrase candidates.
- Produce visual diagnostics (PCFG fragments, CKY charts, POS bigram heatmaps and Sankey flows) and an HTML dashboard per run.

## Quick start

1. Install dependencies (recommended):

   - Python 3.8+
   - pip

   At minimum you will likely need:

       pip install nltk numpy matplotlib networkx plotly

   And download NLTK data:

       python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"

2. Prepare inputs

   - Personal text: plain text file (convention: `personal_<topic>.txt`) placed under `personal_data/` or passed to the pipeline.
   - GPT text: plain text file (convention: `gpt_<topic>.txt`) placed under `gpt_data/` or passed to the pipeline.
   - (Optional) GPT paraphrase candidates: JSONL files under `data/`, e.g. `data/gpt_candidates_<topic>.jsonl`.

3. Run the full POS pipeline for one pair

   Example (single personal / single GPT essay):

       python run_pos_pipeline.py \
         --label alcohol_dbb \
         --personal-text personal_data/personal_dbb.txt \
         --gpt-text gpt_data/gpt_dbb.txt \
         --gpt-cands data/gpt_candidates_dbb.jsonl

   This creates a timestamped run folder under `runs/` with subfolders `models/`, `viz/`, `personal_data/`, `gpt_data/`, a `dashboard.html`, and `log.txt`.

4. Batch mode (one personal file per GPT file, matched by filename prefix)

       python run_pos_pipeline.py \
         --label batch_all \
         --personal-dir personal_data \
         --gpt-dir gpt_data \
         --gpt-cands data/gpt_candidates.jsonl

   In batch mode the script expects `gpt_<topic>.txt` to have a matching `personal_<topic>.txt` in the personal dir.

## Key scripts and entry points

- run_pos_pipeline.py — Main pipeline runner. Performs preprocessing, builds POS trees, estimates PCFGs, computes baselines, builds style stats, rewrites GPT essays, analyzes POS changes, generates visualizations, and writes an HTML dashboard.

- rewrite_POS.py — Standalone script that rewrites a GPT essay toward personal style on a per-sentence basis. Uses the POS PCFGs and style baselines, and can incorporate external GPT paraphrase candidates (JSONL). Example:

      python rewrite_POS.py \
        --mine personal_data/personal_dbb_clean.txt \
        --gpt gpt_data/gpt_dbb_clean.txt \
        --you-grammar models/pcfg_personal_pos.json \
        --gpt-grammar models/pcfg_gpt_pos.json \
        --baselines models/style_baselines.json \
        --cands data/gpt_candidates_dbb.jsonl \
        --out-essay rewritten_gpt_essay_relative_pos.txt \
        --out-log models/pos_change_log.json

-make_POS_trees.py - uses benepar (Berkeley Neural Parser) to create constituency trees whose terminal leaves are POS tags rather than lexical tokens (words). Use the non POS scripts to run on Lexical (word) tokens instead. 

- pcfg_POS.py — Estimate a POS-based PCFG from POS-tree JSONL (converts trees to CNF, collects productions, and writes a JSON mapping from LHS → [ {rhs, prob, log_prob}, ... ]).

- build_style_stats_pos.py — Build POS unigram/bigram and lexical Pw|POS statistics for personal and GPT texts and compute log-odds deltas.

- analyze_pos_changes.py — Aggregate sentence-level POS change logs (from rewrite) and produce plots: top added/removed POS bigrams, heatmaps, transition diagrams, and Sankey flow HTML (if plotly is installed).

Other helper scripts (used by the pipeline): process_sample_text.py, make_POS_trees.py, visualize_cky_chart.py, visualize_graph_pcfg.py, grammar_fingerprint_viz.py, diagnose_sentence_style_pos.py, cky_POS.py, cky_viterbi.py, etc.

## Expected inputs & outputs (per-run)

Inputs (copied to run folder):
- personal raw text - run_dir/personal_data/
- GPT raw text - run_dir/gpt_data/
- optional paraphrase JSONL - run_dir/data/

Produced outputs (under run_dir):
- models/pcfg_personal_pos.json and models/pcfg_gpt_pos.json
- models/style_baselines.json (mu_you_on_you, mu_gpt_on_gpt)
- models/style_stats_pos.json (POS and lexical stats + deltas)
- rewritten_gpt_essay_relative_pos.txt (rewritten essay)
- models/pos_change_log.json (sentence-level POS bigram add/remove records)
- viz/ (PNG/HTML visualizations) and dashboard.html

## Notes and tips

- Preprocessing: `run_pos_pipeline.py` invokes `process_sample_text.py` to generate cleaned text and JSONL. If you skip preprocessing, make sure the expected `*_clean.txt` and `*_clean.jsonl` files exist.

- Matching files in batch mode: When `--gpt-dir` is provided, the pipeline looks for a corresponding `personal_<topic>.txt` (it replaces the `gpt_` prefix with `personal_`).

- Paraphrase JSONL format: rewrite_POS.py accepts JSONL records with either `{"sent_idx": <int>, "cands": [...]} ` or `{ "orig": "<sentence>", "cands": [...] }`. The script matches `orig` strings to tokenized sentences in the GPT essay.

- Missing optional packages (plotly/networkx): visualization steps that rely on these libraries will be skipped with a warning if they are not installed.

- Unparsable sentences: CKY scoring/parsing may fail for some sentences; rewrite scripts apply a strong negative fallback score so unparsable candidates are deprioritized.

## Repo structure (high level)

- run_pos_pipeline.py
- rewrite_POS.py
- pcfg_POS.py
- build_style_stats_pos.py
- analyze_pos_changes.py
- describe_pcfg.py
- process_sample_text.py, make_POS_trees.py, visualize_* scripts, cky_*.py, grammar_fingerprint_viz.py, diagnose_sentence_style_pos.py
- data/ (paraphrase candidates and shared inputs)
- personal_data/ (example personal texts)
- gpt_data/ (example GPT texts)
- models/ (PCFG JSONs, style baselines, stats)
- runs/ (generated experiment runs)
- final_visualizations/, COMPARISON_RESULTS/ (baked outputs included in repo)
