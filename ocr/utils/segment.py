import re
import nltk
import os

nltk.data.path.append(os.getenv("NLTK_DATA", "/usr/local/nltk_data"))

nltk.download("punkt", quiet=True)

from nltk.tokenize import sent_tokenize

def segment_text(text, mode="sentence"):
    if mode == "sentence":
        return sent_tokenize(text)
    elif mode == "paragraph":
        return [p.strip() for p in text.split("\n\n") if len(p.strip()) > 20]