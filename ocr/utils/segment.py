import re
from typing import List, Tuple
import os
import nltk

nltk.data.path.append(os.getenv("NLTK_DATA", "/usr/local/nltk_data"))

nltk.download("punkt", quiet=True)

from nltk.tokenize import sent_tokenize

def segment_text(text, mode="sentence"):
    if mode == "sentence":
        return sent_tokenize(text)
    elif mode == "paragraph":
        return [p.strip() for p in text.split("\n\n") if len(p.strip()) > 20]
    
def segment_pages_to_paragraphs(page_texts: List[str]) -> Tuple[List[str], List[int]]:
    segments, seg2page = [], []
    for i, ptxt in enumerate(page_texts):
        paras = [p.strip() for p in ptxt.split("\n\n") if p.strip()]
        if not paras and ptxt.strip():
            paras = [ptxt.strip()]
        for para in paras:
            segments.append(para)
            seg2page.append(i)
    return segments, seg2page

def segment_pages_to_sentences(pages: List[str]) -> Tuple[List[str], List[int]]:
    """
    Split each page's text into sentences using NLTK.
    Returns:
      segments: list of sentence strings
      seg2page: list of page indices aligned with segments
    """
    segments: List[str] = []
    seg2page: List[int] = []

    for page_idx, page_text in enumerate(pages):
        if not page_text.strip():
            continue

        # NLTK's sentence tokenizer
        sents = nltk.sent_tokenize(page_text)
        for s in sents:
            s_clean = s.strip()
            if not s_clean:
                continue
            segments.append(s_clean)
            seg2page.append(page_idx)

    return segments, seg2page