'''
PROCESS SAMPLE TEXT:
Clean and segment essays into sents
Output:
    JSONL with one sent / line (mine_clean.jsonl)
    Cleaned .txt file per essay (DOCID_clean.txt)
'''

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
import argparse  # <-- NEW
from typing import Iterator
from nltk import word_tokenize, pos_tag
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

#PREPROCESS

def normalize_unicode(s: str) -> str:
    #normalize unicode chars
    s = unicodedata.normalize("NFC", s)
    #normalize whitespace
    s = re.sub(r"[ \t]+", " ", s)
    #normalize dashes
    s = re.sub(r"[–—\-]{2,}", "—", s)  #long dashes to em dashes
    s = re.sub(r" ?— ?", " — ", s)     #pad em dashes 
    #normalize ellipses
    s = s.replace("…", "...")
    return s.strip()

#normalize urls and code blocks 
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
CODE_LINE = re.compile(r"^\s*[`~]{3,}.*$")

#citation patterns to strip (in text ref)
CITATION_RE = re.compile(
    r"""\(
        [A-Z][A-Za-z\-]+                     # author
        (?:\s+(?:et\ al\.?|[A-Z][A-Za-z\-]+))*  # addl. authors or 'et al.'
        \s*,\s*
        \d{4}                                #yr
        [a-z]?                               #letter suffix 2008a, 2008b
        (?:\s*;\s*\d{4}[a-z]?)?              #possible 2nd year
    \)""",
    re.VERBOSE
)

def clean_line(line: str) -> str:
    """rm code blocks, URLs, and citation parentheticals from line"""
    if CODE_LINE.match(line):
        return ""
    line = URL_RE.sub("", line)
    #drop full lines when just citations or mostly citations
    if CITATION_RE.search(line):
        line = CITATION_RE.sub("", line)
    return line

def get_sent(txt: str) -> Iterator[str]:
    """Use NLTK to get sents btwn 4-80 toks"""
    for s in sent_tokenize(txt):
        sent = s.strip()
        if 4 <= len(sent.split()) <= 80:
            yield sent

def process_file(path: pathlib.Path, doc_id: str) -> list[dict]:
    '''
    Process raw txt file into lst of sent records + metadata
    - doc_id: file stem
    - para_id: idx of paragraph within the doc
    - sent_id: idx of sent within the paragraph
    - text: the sent txt
    '''
    raw = path.read_text(encoding="utf-8", errors="ignore")

    #normalize unicode/spacing/dashes/ellipses
    raw = normalize_unicode(raw)

    #line-lvl cleaning
    lines = [clean_line(l) for l in raw.splitlines()]

    #drop useless lines ex headers
    lines = [l for l in lines if not re.match(r"^\s*(Figure|Table|References)\b", l)]

    #rejoin into single blob of text with newlines
    cleaned = "\n".join(lines)

    out: list[dict] = []
    para_id = 0

    #split into paragraphs: two or more newlines = paragraph boundary
    for para in re.split(r"\n{2,}", cleaned):
        para = para.strip()
        if not para:
            continue
        para_id += 1
        sent_id = 0

        #use nltk to get sents from paragraph
        for sent in get_sent(para):
            sent_id += 1
            out.append(
                {
                    "doc_id": doc_id,
                    "para_id": para_id,
                    "sent_id": sent_id,
                    "text": sent,
                }
            )

    #rm exact sent duplicates in doc
    seen: set[str] = set()
    unique: list[dict] = []
    for r in out:
        key = r["text"]
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)

    return unique

def main(
    in_dir: str,
    out_path: str,
) -> None:
    """
    parse all .txt files in in_dir, process into sent-lvl records
    write to JSONL (mine_clean.jsonl), write cleaned .txt file per essay

    JSONL: 1 line per sent with doc_id, para_id, sent_id, text
    Cleaned .txt per doc: DOCID_clean.txt with all cleaned sents
    """

    in_dir_path = pathlib.Path(in_dir)
    records: list[dict] = []
    #per-essay cleaned txt outputs
    doc_to_sentences: dict[str, list[str]] = {}

    #process txt files in dir
    for p in sorted(in_dir_path.glob("*.txt")):
        if p.suffix.lower() not in {".txt"}:
            continue

        doc_id = p.stem
        doc_records = process_file(p, doc_id=doc_id)
        records.extend(doc_records)

        #collect sents in doc order for per-essay .txt output
        doc_records_sorted = sorted(
            doc_records, key=lambda r: (r["para_id"], r["sent_id"])
        )
        doc_to_sentences[doc_id] = [r["text"] for r in doc_records_sorted]

    out_file = pathlib.Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    #write sent-lvl JSONL
    with out_file.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} sentences to {out_path}")

    #write 1 cleaned .txt file / essay/doc
    base_dir = out_file.parent
    for doc_id, sents in doc_to_sentences.items():
        clean_txt_path = base_dir / f"{doc_id}_clean.txt"
        with clean_txt_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(sents) + "\n")
        print(f"Wrote cleaned essay for {doc_id} to {clean_txt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean and segment essays into sent-lvl JSONL and per-essay cleaned .txt file"
    )
    parser.add_argument(
        "--in-dir",
        type=str,
        required=True,
        help="Input dir with raw .txt essay files",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        required=True,
        help="Output JSONL path (sent-lvl cleaned data)",
    )

    args = parser.parse_args()
    main(in_dir=args.in_dir, out_path=args.out_path)