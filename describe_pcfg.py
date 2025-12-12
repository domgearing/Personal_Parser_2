import argparse
import json
import pathlib
from typing import Dict, List, Tuple

#tag desc from Penn Treebank doc

PHRASE_TAG_DESCRIPTIONS = {
    "S": "Clause / sentence",
    "SBAR": "Subordinate clause (introduced by 'that', 'because', etc.)",
    "SBARQ": "Direct question with fronted wh-word",
    "SINV": "Inverted declarative sentence",
    "SQ": "Inverted yes/no question",
    "NP": "Noun phrase",
    "VP": "Verb phrase",
    "PP": "Prepositional phrase",
    "ADJP": "Adjective phrase",
    "ADVP": "Adverb phrase",
    "PRT": "Particle",
    "CONJP": "Conjunction phrase",
    "INTJ": "Interjection",
    "LST": "List marker",
    "NAC": "Not a Constituent / fragment",
    "NX": "Complex noun-like phrase",
    "WHNP": "Wh-noun phrase (who, what, which...)",
    "WHPP": "Wh-prepositional phrase (in which, to whom...)",
    "WHADJP": "Wh-adjective phrase (how big...)",
    "WHADVP": "Wh-adverb phrase (why, how...)",
    "QP": "Quantifier phrase",
    "X": "Unknown/other phrase",
    "ROOT": "Root of the parse tree",
}

POS_TAG_DESCRIPTIONS = {
    "CC": "Coordinating conjunction (and, or, but)",
    "CD": "Cardinal number",
    "DT": "Determiner (the, a, this)",
    "EX": "Existential 'there'",
    "FW": "Foreign word",
    "IN": "Preposition/subordinating conjunction (in, on, because)",
    "JJ": "Adjective",
    "JJR": "Adjective, comparative (bigger)",
    "JJS": "Adjective, superlative (biggest)",
    "LS": "List item marker",
    "MD": "Modal (can, will, should)",
    "NN": "Noun, singular or mass",
    "NNS": "Noun, plural",
    "NNP": "Proper noun, singular",
    "NNPS": "Proper noun, plural",
    "PDT": "Predeterminer (all, many)",
    "POS": "Possessive ending ('s)",
    "PRP": "Personal pronoun (I, you, we)",
    "PRP$": "Possessive pronoun (my, your, our)",
    "RB": "Adverb (quickly, not)",
    "RBR": "Adverb, comparative (faster)",
    "RBS": "Adverb, superlative (fastest)",
    "RP": "Particle (up, off)",
    "SYM": "Symbol",
    "TO": "Infinitive 'to'",
    "UH": "Interjection (uh, well)",
    "VB": "Verb, base form",
    "VBD": "Verb, past tense",
    "VBG": "Verb, gerund/present participle",
    "VBN": "Verb, past participle",
    "VBP": "Verb, non-3sg present",
    "VBZ": "Verb, 3sg present",
    "WDT": "Wh-determiner (which, that)",
    "WP": "Wh-pronoun (who, what)",
    "WP$": "Possessive wh-pronoun (whose)",
    "WRB": "Wh-adverb (where, when, how)",
    ",": "Comma",
    ".": "Sentence-final punctuation (. ! ?)",
    ":": "Colon, semi-colon, dash",
    "''": "Closing quote",
    "``": "Opening quote",
    "-LRB-": "Left round bracket '('",
    "-RRB-": "Right round bracket ')'",
}

def load_pcfg(path: str) -> Dict[str, list]:
    p = pathlib.Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def classify_symbol(sym: str) -> Tuple[str, str]:
    """
    classify symbol as 'phrase', 'pos', or 'unknown',
    return (category, description).
    """
    if sym in PHRASE_TAG_DESCRIPTIONS:
        return "phrase", PHRASE_TAG_DESCRIPTIONS[sym]
    if sym in POS_TAG_DESCRIPTIONS:
        return "pos", POS_TAG_DESCRIPTIONS[sym]
    # multi-letter all-caps - phrase-like; 2-letter all-caps - POS-like
    if len(sym) > 2:
        return "phrase?", "Likely phrase/category (not in table)"
    else:
        return "pos?", "Likely POS tag (not in table)"

def describe_pcfg(pcfg: Dict[str, list]) -> None:
    # basic stats
    num_lhs = len(pcfg)
    num_rules = sum(len(rules) for rules in pcfg.values())
    print(f"PCFG summary")
    print(f"  Nonterminals (LHS): {num_lhs}")
    print(f"  Total rules:        {num_rules}")
    print()

    #build rows for each NT
    rows = []
    for lhs, rules in sorted(pcfg.items()):
        rule_count = len(rules)
        cat, desc = classify_symbol(lhs)
        rows.append((lhs, cat, rule_count, desc))

    #pretty-print to console
    print(f"{'SYM':<8} {'TYPE':<7} {'#RULES':<7} DESCRIPTION")
    print("-" * 80)
    for lhs, cat, rule_count, desc in rows:
        print(f"{lhs:<8} {cat:<7} {rule_count:<7} {desc}")

def export_symbol_table(
    pcfg: Dict[str, list],
    csv_path: str = "outputs/nonterminals.csv",
    md_path: str = "outputs/nonterminals.md",
) -> None:
    out_dir = pathlib.Path(csv_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for lhs, rules in sorted(pcfg.items()):
        rule_count = len(rules)
        cat, desc = classify_symbol(lhs)
        rows.append((lhs, cat, rule_count, desc))

    #CSV
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("symbol,type,num_rules,description\n")
        for lhs, cat, rule_count, desc in rows:
            # escape quotes in description
            safe_desc = desc.replace('"', '""')
            f.write(f'"{lhs}","{cat}",{rule_count},"{safe_desc}"\n')

    #mkd table
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| Symbol | Type | #Rules | Description |\n")
        f.write("|--------|------|--------|-------------|\n")
        for lhs, cat, rule_count, desc in rows:
            f.write(f"| `{lhs}` | {cat} | {rule_count} | {desc} |\n")

    print(f"Wrote symbol table to {csv_path} and {md_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Summarize PCFG: rule counts, tag desc, and export tables"
    )
    parser.add_argument(
        "--grammar",
        type=str,
        required=True,
        help="Path to PCFG JSON",
    )
    parser.add_argument(
        "--csv-out",
        type=str,
        default=None,
        help="Optl. path to write CSV of NT's (default: skip)",
    )
    parser.add_argument(
        "--md-out",
        type=str,
        default=None,
        help="Optl. path to write MaKD table (default: skip)",
    )
    args = parser.parse_args()

    pcfg = load_pcfg(args.grammar)
    describe_pcfg(pcfg)
    if args.csv_out and args.md_out:
        export_symbol_table(pcfg, csv_path=args.csv_out, md_path=args.md_out)
    elif args.csv_out:
        export_symbol_table(pcfg, csv_path=args.csv_out, md_path=args.csv_out + ".md")
    elif args.md_out:
        export_symbol_table(pcfg, csv_path=args.md_out.replace(".md", ".csv"), md_path=args.md_out)

if __name__ == "__main__":
    main()
