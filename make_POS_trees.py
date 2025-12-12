"""
IMPORT NECESSARY PACKAGES AND LIBRARIES
"""

import numpy as np
import pandas as pd 
import scipy
import sklearn as sk
import tqdm
import regex
import pydantic
import re 
import os 
import matplotlib as plt
import seaborn as sns 
import nltk
import spacy
import benepar
import typing 
import unicodedata
import json
import pathlib
import typing
import sys
import argparse  # <-- added
from typing import Iterator, Optional
from nltk import word_tokenize
from nltk.corpus import stopwords   
from collections import Counter
from nltk import sent_tokenize
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tokenize import *

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
benepar.download('benepar_en3')

### benepar_en3: Berkeley Neural Parser English model, v3     ###
### using this because i don't have enough personal writing   ###
### samples to train my own parser                            ###


# load benepar model 
def load_benepar_model(model_name: str = 'benepar_en3') -> benepar.Parser:
    """
    returns: parser
    accepts: token lists 
    outputs: NLTK trees
    """
    try:
        parser = benepar.Parser(model_name)
    except ValueError as e:
        print(
            f"Error loading benepar model '{model_name}'. "
            f"Make sure you've downloaded it with:\n"
            f"  python -m benepar.download '{model_name}'",
            file=sys.stderr,
        )
        raise e
    return parser


def parse_sentence(parser: benepar.Parser, sent_text: str) -> Optional[dict]:
    """
    Tokenize and parse a single sentence string using benepar parser.

    Converts leaves to **POS tags**.

    Returns dict with:
        tokens: list of POS tags
        tree_str: bracketed tree string with POS leaves
    """
    try:
        tokens = word_tokenize(sent_text)
        pos_tokens = [tag for _, tag in nltk.pos_tag(tokens)]
        tree = parser.parse(tokens)
    except Exception as e:
        print(f"[WARN] Parse failed for sentence: {sent_text!r}\n  Error: {e}", 
              file=sys.stderr)
        return None

    # convert leaves to POS tags
    for subtree in tree.subtrees():
        if len(subtree) == 1 and isinstance(subtree[0], str):
            subtree[0] = subtree.label()

    tree_str = tree.pformat(margin=1_000_000)

    return {
        "tokens": pos_tokens,
        "tree": tree_str
    }


def process_clean_file(
        in_path: str,
        out_path: str,
        max_sentences: Optional[int] = None) -> None:
    """
    1. Read sentences from a cleaned JSONL file
    2. Parse each sentence into constituency tree using benepar
    3. Output a JSONL with:
        doc_id, para_id, sent_id, text, tokens, tree
    """
    parser = load_benepar_model("benepar_en3")

    in_file = pathlib.Path(in_path)
    out_file = pathlib.Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    num_in = 0
    num_parsed = 0

    with in_file.open("r", encoding="utf-8") as fin, \
         out_file.open("w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            num_in += 1
            if max_sentences is not None and num_in > max_sentences:
                break

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] Skipping JSON line: {line!r}", file=sys.stderr)
                continue

            text = record.get("text", "")
            if not text:
                continue

            parsed = parse_sentence(parser, text)
            if parsed is None:
                continue

            num_parsed += 1

            out_record = {
                "doc_id": record.get("doc_id"),
                "para_id": record.get("para_id"),
                "sent_id": record.get("sent_id"),
                "text": text,
                "tokens": parsed["tokens"],
                "tree": parsed["tree"],
            }

            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")

    print(f"Read {num_in} sentences from {in_path}")
    print(f"Successfully parsed {num_parsed} sentences to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Parse cleaned txt into POS-based constituency trees")
    parser.add_argument("--in-path", type=str, required=True, help="Input JSONL containing cleaned sents")
    parser.add_argument("--out-path", type=str, required=True, help="Output JSONL to store parsed POS trees")
    parser.add_argument("--max-sentences", type=int, default=None, help="Optional limit on # of sents to parse")

    args = parser.parse_args()

    process_clean_file(
        in_path=args.in_path,
        out_path=args.out_path,
        max_sentences=args.max_sentences
    )


if __name__ == "__main__":
    main()