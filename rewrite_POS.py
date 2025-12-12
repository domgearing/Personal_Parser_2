from typing import List, Tuple, Optional, Dict
import argparse
import pathlib
import json

from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

from cky_POS import PCFG, parse_and_style_score

SUPER_NEG = -1e6  # backup style score for unparsable sents

STRUCTURAL_CHANGES = []  #glob log of POS-struct changes


def load_baselines(path: str = "models/style_baselines.json"):
    p = pathlib.Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    return data["mu_you_on_you"], data["mu_gpt_on_gpt"]


#cand gen. for single sent (heuristic based)

def generate_candidates(sent: str) -> List[str]:
    """
    cand generator for single sent - backup option
    encodes 'GPT-ish -> you-ish' preferences based on style diffs:
        GPT-ish discourse markers -> simpler forms
        WHPP / QP-ish formal phrases -> simpler alternatives
        A rhetorical/question-like variant for some 'This...' sents
    """
    cands = [sent]

    #discourse marker + formal opens
    replacements = [
        ("In conclusion,", "Overall,"),
        ("In conclusion ,", "Overall ,"),
        ("In conclusion ", "Overall "),
        ("in conclusion,", "overall,"),
        ("in conclusion ,", "overall ,"),
        ("However,", "But"),
        ("However ,", "But"),
        ("however,", "but"),
        ("however ,", "but"),
        ("Moreover,", "Also"),
        ("Moreover ,", "Also"),
        ("moreover,", "also"),
        ("moreover ,", "also"),
        ("Therefore,", "So"),
        ("Therefore ,", "So"),
        ("therefore,", "so"),
        ("therefore ,", "so"),
    ]

    #WHPP / QP-ish pattern (from compare_styles)
    replacements += [
        ("to which", "that"),
        ("to whom", "who"),
        ("to what extent", "how"),
        ("in many cases", "often"),
        ("in many ways", "often"),
        ("in a variety of ways", "in different ways"),
        ("in a number of ways", "in different ways"),
    ]

    for old, new in replacements:
        if old in sent:
            cands.append(sent.replace(old, new))

    #add rhetorical/question-like bit for 'This ...' sents
    stripped = sent.lstrip()
    if stripped.startswith("This ") or stripped.startswith("this "):
        #turn "This pattern suggests that X"
        #into "What this pattern suggests is that X"
        words = stripped.split()
        if len(words) >= 3:
            noun = words[1]
            #everything after noun
            tail = " ".join(words[2:])
            rhetorical = f"What this {noun} suggests is that {tail}"
            cands.append(rhetorical)

    #deduplicate while pres. order
    seen = set()
    deduped = []
    for c in cands:
        if c not in seen:
            seen.add(c)
            deduped.append(c)

    return deduped


#load GPT paraphrase cands

def load_paraphrase_candidates(path: str, gpt_essay_text: str) -> Dict[int, List[str]]:
    """
    Load paraphrase cands from JSONL
    match 'orig' to GPT essay's sents to get sent idx
    """
    out: Dict[int, List[str]] = {}
    p = pathlib.Path(path)
    if not p.exists():
        print(f"[WARN] No paraphrase cand file at {path}; skipping GPT cands ")
        return out

    #make map from sent text -> idx for GPT essay
    essay_sents = [s.strip() for s in sent_tokenize(gpt_essay_text) if s.strip()]
    sent_to_idx: Dict[str, int] = {}
    for i, s in enumerate(essay_sents):
        #use string key; if duplicates exist, keep first
        sent_to_idx.setdefault(s, i)

    count_sents = 0
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            #if has explicit idx
            if "sent_idx" in rec:
                idx = rec["sent_idx"]
                cands = rec.get("cands", [])
                if idx not in out:
                    out[idx] = []
                    count_sents += 1
                out[idx].extend(cands)
                continue

            #if text-based ("orig")
            if "orig" in rec:
                orig = rec["orig"].strip()
                cands = rec.get("cands", [])
                idx = sent_to_idx.get(orig)
                if idx is None:
                    continue
                if idx not in out:
                    out[idx] = []
                    count_sents += 1
                out[idx].extend(cands)

    print(f"[INFO] Loaded paraphrases for {count_sents} sentences from {path}")
    return out


#rel. style scoring (you vs GPT)

def style_score_sentence_relative(
    sent: str,
    g_you: PCFG,
    g_gpt: PCFG,
    mu_you_on_you: float,
    mu_gpt_on_gpt: float,
) -> Tuple[float, Optional[float], Optional[float]]:
    """
    return normalized rel. style score for sent:
    style_you_norm = style_you - mu_you_on_you
    style_gpt_norm = style_gpt - mu_gpt_on_gpt
    style_rel = style_you_norm - style_gpt_norm

    Positive  -> closer to personal training distribution than GPT's
    Negative  -> closer to GPT's training distribution than personal

    returns (style_you_norm, style_gpt_norm) for debugging
    """
    #parse_and_style_score: (logp, tree, avg_logP_per_rule, maybe_extra)
    _, _, style_you, _ = parse_and_style_score(sent, g_you)
    _, _, style_gpt, _ = parse_and_style_score(sent, g_gpt)

    if style_you is None or style_gpt is None:
        return SUPER_NEG, None, None

    style_you_norm = style_you - mu_you_on_you
    style_gpt_norm = style_gpt - mu_gpt_on_gpt
    style_rel = style_you_norm - style_gpt_norm

    return style_rel, style_you_norm, style_gpt_norm


def score_essay_relative(
    text: str,
    g_you: PCFG,
    g_gpt: PCFG,
    mu_you_on_you: float,
    mu_gpt_on_gpt: float,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    get overall normalized rel. style for essay
    avg_rel = mean over sentences of (style_you_norm - style_gpt_norm)

    return
    avg_you_norm = avg normalized style_you across sentences
    avg_gpt_norm = avg normalized style_gpt across sentences

    return (None, None, None) if unparsable
    """
    sents = [s.strip() for s in sent_tokenize(text) if s.strip()]
    if not sents:
        return None, None, None

    rel_scores: List[float] = []
    you_scores: List[float] = []
    gpt_scores: List[float] = []

    for s in sents:
        rel, sy_norm, sg_norm = style_score_sentence_relative(
            s, g_you, g_gpt, mu_you_on_you, mu_gpt_on_gpt
        )
        if rel == SUPER_NEG or sy_norm is None or sg_norm is None:
            continue
        rel_scores.append(rel)
        you_scores.append(sy_norm)
        gpt_scores.append(sg_norm)

    if not rel_scores:
        return None, None, None

    avg_rel = sum(rel_scores) / len(rel_scores)
    avg_you = sum(you_scores) / len(you_scores)
    avg_gpt = sum(gpt_scores) / len(gpt_scores)

    return avg_rel, avg_you, avg_gpt


def split_into_sentences(text: str) -> List[str]:
    """
    split essay into sents - NLTK sent_tokenize
    """
    return [s.strip() for s in sent_tokenize(text) if s.strip()]


#POS structural comparison

def pos_sequence(sent: str) -> List[str]:
    """Return POS tag sequence for a sentence."""
    tokens = word_tokenize(sent)
    tagged = nltk.pos_tag(tokens)
    return [pos for _, pos in tagged]


def pos_bigrams(pos_seq: List[str]) -> List[Tuple[str, str]]:
    return list(zip(pos_seq, pos_seq[1:])) if len(pos_seq) > 1 else []


def explain_pos_changes(orig: str, new: str):
    """
    Compute POS seq and added/rem POS bigrams btwn orig and new sent
    """
    pos_o = pos_sequence(orig)
    pos_n = pos_sequence(new)

    big_o = set(pos_bigrams(pos_o))
    big_n = set(pos_bigrams(pos_n))

    added = sorted(list(big_n - big_o))
    removed = sorted(list(big_o - big_n))

    return pos_o, pos_n, added, removed


def rewrite_essay_to_your_style(
    gpt_essay: str,
    g_you: PCFG,
    g_gpt: PCFG,
    mu_you_on_you: float,
    mu_gpt_on_gpt: float,
    paraphrases: Optional[Dict[int, List[str]]] = None,
) -> str:
    """
    rewrite GPT essay sent by sent using normalized rel. style
    for each sent:
        generate heuristic cands
        add GPT paraphrase cands
        score by normalized rel. style (you_norm - gpt_norm)
        pick best

    prints for each sent:
        orig vs final
        style_rel before vs after 
        high lvl grammatical changes through POS patterns / bigrams
    """
    if paraphrases is None:
        paraphrases = {}

    sents = split_into_sentences(gpt_essay)
    rewritten_sents: List[str] = []
    changed = 0

    print("\n=== PER-SENTENCE CHANGES (STRUCTURAL / POS-LEVEL) ===\n")

    for idx, s in enumerate(sents):
        #style score of orig sent
        orig_rel, _, _ = style_score_sentence_relative(
            s, g_you, g_gpt, mu_you_on_you, mu_gpt_on_gpt
        )

        #start with heuristic cands (includes orig sent)
        cand_list = generate_candidates(s)

        #GPT paraphrase cands for this sent idx
        gpt_cands = paraphrases.get(idx, [])
        cand_list.extend(gpt_cands)

        #Deduplicate while preserving order, make sure orig present
        seen = set()
        all_cands = []
        for c in cand_list:
            if c not in seen:
                seen.add(c)
                all_cands.append(c)
        if s not in seen:
            all_cands.insert(0, s)  #ensure orig included

        #score cands
        all_scored: List[Tuple[str, float]] = []
        for cand in all_cands:
            rel, _, _ = style_score_sentence_relative(
                cand, g_you, g_gpt, mu_you_on_you, mu_gpt_on_gpt
            )
            all_scored.append((cand, rel))

        #choose best
        best_sent, best_rel = max(all_scored, key=lambda x: x[1])

        #count change
        if best_sent != s:
            changed += 1

        #compute POS changes (even if unchanged)
        pos_o, pos_n, added, removed = explain_pos_changes(s, best_sent)

        #log it glob
        STRUCTURAL_CHANGES.append({
            "sent_idx": idx,
            "orig": s,
            "new": best_sent,
            "added": added,
            "removed": removed
        })

        #print explanation
        print(f"Sentence {idx}:")
        print(f"  ORIGINAL : {s}")
        print(f"  REWRITTEN: {best_sent}")
        print(f"  style_rel(orig) = {orig_rel:.3f}, style_rel(new) = {best_rel:.3f}, Δ = {best_rel - orig_rel:.3f}")

        print("  POS pattern (orig):", " ".join(pos_o))
        print("  POS pattern (new) : ", " ".join(pos_n))

        if added or removed:
            print("  POS bigram changes:")
            if added:
                print("    Added:")
                for a in added:
                    print(f"      {a[0]} → {a[1]}")
            if removed:
                print("    Removed:")
                for r in removed:
                    print(f"      {r[0]} → {r[1]}")
        else:
            print("  No structural POS differences detected.")
        #note whether best came from GPT candidates or heuristic/original
        source = "GPT paraphrases" if best_sent in gpt_cands else "heuristic rewrite/original"
        print(f"  Source of chosen rewrite: {source}\n")

        #append final vers. of this sent
        rewritten_sents.append(best_sent)

    print(f"[INFO] Rewrote {changed} out of {len(sents)} sentences.")
    return " ".join(rewritten_sents)


# file I/O + CLI

def read_text_file(path: str) -> str:
    p = pathlib.Path(path)
    return p.read_text(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Score, rewrite GPT essay toward personal style"
    )
    parser.add_argument(
        "--mine",
        type=str,
        required=True,
        help="Path to txt file with personal essay",
    )
    parser.add_argument(
        "--gpt",
        type=str,
        required=True,
        help="Path to txt file with GPT essay to be rewritten",
    )
    parser.add_argument(
        "--cands",
        type=str,
        default=None,
        help="path to JSONL file of GPT paraphrase cands",
    )
    parser.add_argument(
        "--you-grammar",
        type=str,
        default="models/pcfg_personal_pos.json",
        help="Path to POS-based PCFG JSON for personal style",
    )
    parser.add_argument(
        "--gpt-grammar",
        type=str,
        default="models/pcfg_gpt_pos.json",
        help="Path to POS-based PCFG JSON for GPT style",
    )
    parser.add_argument(
        "--baselines",
        type=str,
        default="models/style_baselines.json",
        help="Path to style baselines JSON",
    )
    parser.add_argument(
        "--out-essay",
        type=str,
        default="rewritten_gpt_essay_relative_pos.txt",
        help="Path to save rewritten GPT essay",
    )
    parser.add_argument(
        "--out-log",
        type=str,
        default="models/pos_change_log.json",
        help="Path to save POS structl change log as JSON",
    )

    args = parser.parse_args()

    #load both grammars
    g_you = PCFG(
        start_symbol="S",
        grammar_path=args.you_grammar,
        allow_oov_lexical=True,
    )
    g_gpt = PCFG(
        start_symbol="S",
        grammar_path=args.gpt_grammar,
        allow_oov_lexical=True,
    )

    #load normalization baselines
    mu_you_on_you, mu_gpt_on_gpt = load_baselines(args.baselines)

    #read essays
    my_essay = read_text_file(args.mine)
    gpt_essay = read_text_file(args.gpt)

    #load paraphrase candidates
    paraphrases: Dict[int, List[str]] = {}
    if args.cands is not None:
        paraphrases = load_paraphrase_candidates(args.cands, gpt_essay)

    #score personal essay (relative, normalized)
    my_rel, my_you_norm, my_gpt_norm = score_essay_relative(
        my_essay, g_you, g_gpt, mu_you_on_you, mu_gpt_on_gpt
    )
    print("=== YOUR ESSAY ===")
    print(f"Length (chars): {len(my_essay)}")
    print("Average normalized relative style (you - GPT):", my_rel)
    print("Average normalized logP_you per rule        :", my_you_norm)
    print("Average normalized logP_gpt per rule        :", my_gpt_norm)
    print("Interpretation: positive = closer to your training distribution than GPT's.\n")

    #score orig GPT essay
    gpt_rel_before, gpt_you_norm_before, gpt_gpt_norm_before = score_essay_relative(
        gpt_essay, g_you, g_gpt, mu_you_on_you, mu_gpt_on_gpt
    )
    print("=== ORIGINAL GPT ESSAY ===")
    print(f"Length (chars): {len(gpt_essay)}")
    print("Average normalized relative style (you - GPT):", gpt_rel_before)
    print("Average normalized logP_you per rule        :", gpt_you_norm_before)
    print("Average normalized logP_gpt per rule        :", gpt_gpt_norm_before)
    print()

    #rewrite GPT essay (per-sentence POS / structure explanations)
    rewritten = rewrite_essay_to_your_style(
        gpt_essay, g_you, g_gpt, mu_you_on_you, mu_gpt_on_gpt, paraphrases=paraphrases
    )

    #re-score rewritten essay
    gpt_rel_after, gpt_you_norm_after, gpt_gpt_norm_after = score_essay_relative(
        rewritten, g_you, g_gpt, mu_you_on_you, mu_gpt_on_gpt
    )
    print("=== REWRITTEN GPT ESSAY (STYLE-ADAPTED) ===")
    print(f"Length (chars): {len(rewritten)}")
    print("Average normalized relative style (you - GPT):", gpt_rel_after)
    print("Average normalized logP_you per rule        :", gpt_you_norm_after)
    print("Average normalized logP_gpt per rule        :", gpt_gpt_norm_after)

    #compare rel. style shift
    if gpt_rel_before is not None and gpt_rel_after is not None:
        delta = gpt_rel_after - gpt_rel_before
        print("=== RELATIVE STYLE SHIFT (toward you, away from GPT) ===")
        print(f"Before (you - GPT): {gpt_rel_before}")
        print(f"After  (you - GPT): {gpt_rel_after}")
        print(f"Δ (after - before): {delta:.4f}")
        if delta > 0:
            print("-> Rewritten GPT essay is more you-ish relative to GPT.")
        elif delta < 0:
            print("-> Rewriting accidentally moved it toward GPT's style.")
        else:
            print("-> No net change in relative style score.")
    else:
        print("[INFO] Could not compute essay scores")

    #save rewritten essay
    out_path = pathlib.Path(args.out_essay)
    out_path.write_text(rewritten, encoding="utf-8")
    print(f"\nRewritten essay saved to: {out_path}")
    
    #save structl change log
    log_path = pathlib.Path(args.out_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(STRUCTURAL_CHANGES, indent=2))
    print(f"[INFO] Saved POS structural change log to {log_path}")


if __name__ == "__main__":
    main()
