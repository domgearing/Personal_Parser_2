import argparse
import json
import pathlib
from typing import Optional

from nltk.tokenize import sent_tokenize

from cky_POS import PCFG, parse_and_style_score


def compute_baseline(corpus_path: str, grammar: PCFG) -> Optional[float]:
    """
    Compute mean avg_logP_per_rule for all parsable sents in corpus under grammar
    """
    p = pathlib.Path(corpus_path)
    text = p.read_text(encoding="utf-8")

    sents = [s.strip() for s in sent_tokenize(text) if s.strip()]
    scores = []

    for s in sents:
        _, _, style, _ = parse_and_style_score(s, grammar)
        if style is not None:
            scores.append(style)

    if not scores:
        return None

    return sum(scores) / len(scores)


def main():
    parser = argparse.ArgumentParser(
        description="Compute style baselines for personal, GPT PCFGs"
    )
    parser.add_argument(
        "--mine-train",
        type=str,
        required=True,
        help="Path to training corpus for YOUR grammar",
    )
    parser.add_argument(
        "--gpt-train",
        type=str,
        required=True,
        help="Path to training corpus for GPT grammar",
    )
    parser.add_argument(
        "--gpt_grammar",
        type=str,
        required=True,
        help="Path to PCFG model for GPT",
    )
    parser.add_argument(
        "--personal_grammar",
        type=str,
        required=True,
        help="Path to PCFG model for your personal style",
    )
    parser.add_argument(
        "--out-baselines",
        type=str,
        default="models/style_baselines.json",
        help="Path to save baselines JSON (mu_you_on_you, mu_gpt_on_gpt)",
    )

    args = parser.parse_args()

    #load grammars
    g_you = PCFG(
        start_symbol="S",
        grammar_path=args.personal_grammar,
        allow_oov_lexical=True,
    )
    g_gpt = PCFG(
        start_symbol="S",
        grammar_path=args.gpt_grammar,
        allow_oov_lexical=True,
    )

    print("Computing baseline for YOUR grammar on YOUR corpus...")
    mu_you = compute_baseline(args.mine_train, g_you)
    print("  mu_you_on_you:", mu_you)

    print("\nComputing baseline for GPT grammar on GPT corpus...")
    mu_gpt = compute_baseline(args.gpt_train, g_gpt)
    print("  mu_gpt_on_gpt:", mu_gpt)

    out = {
        "mu_you_on_you": mu_you,
        "mu_gpt_on_gpt": mu_gpt,
    }

    out_path = pathlib.Path(args.out_baselines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved baselines to {out_path}")


if __name__ == "__main__":
    main()
