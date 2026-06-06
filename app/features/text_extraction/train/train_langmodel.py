import os
import json
import pickle
import re

from collections import defaultdict, Counter

from app.features.text_extraction.config import parameters
from app.features.text_extraction.models.langmodel import DawgNode


def _load_corpus(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().lower()

def _build_bigrams(text):
    chars = [c for c in text if c.isalpha() or c == " "]
    counts = defaultdict(lambda: defaultdict(float))
    totals = defaultdict(float)

    for a, b in zip(chars, chars[1:]):
        if a == " " or b == " ":
            continue
        counts[a][b] += 1.0
        totals[a] += 1.0

    alphabet = "abcdefghijklmnopqrstuvwxyz"
    probs = {}
    for a in alphabet:
        probs[a] = {}
        total = totals[a] + parameters["langmodel"]["smoothing"] * len(alphabet)
        for b in alphabet:
            probs[a][b] = (counts[a][b] + parameters["langmodel"]["smoothing"]) / total

    return probs

def _build_dawg_from_word_frequencies(word_frequencies, min_freq=2):
    root = DawgNode()
    
    filtered_words = [word for word, count in word_frequencies.items() if count >= min_freq]
    
    
    for word in filtered_words:
        node = root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = DawgNode()
            node = node.children[ch]
        node.is_end = True

    return root


def train():
    os.makedirs(parameters["train_dirs"]["models"], exist_ok=True)

    if not os.path.exists(parameters["train_dirs"]["corpus"]):
        return

    print("Loading corpus...")
    text = _load_corpus(parameters["train_dirs"]["corpus"])
    
    all_words = re.findall(r"[a-z]+", text)
    word_frequencies = Counter(all_words)
    
    total_unique_words = len(word_frequencies)
    total_words = len(all_words)
    
    print(f"  {len(text):,} characters")
    print(f"  {total_words:,} total words")
    print(f"  {total_unique_words:,} unique words")
    
    words_once = sum(1 for count in word_frequencies.values() if count == 1)
    print(f"  {words_once:,} words occur only once")
    print(f"  {total_unique_words - words_once:,} words occur 2+ times")

    print("Building bigram table...")
    bigrams = _build_bigrams(text)
    with open(parameters["langmodel"]["bigram_path"], "w", encoding="utf-8") as f:
        json.dump(bigrams, f)
    print(f"  Saved: {parameters['langmodel']['bigram_path']}")

    print("Building DAWG...")
    dawg = _build_dawg_from_word_frequencies(word_frequencies, parameters["langmodel"]["min_word_frequency"])
    with open(parameters["langmodel"]["dawg_path"], "wb") as f:
        pickle.dump(dawg, f)
    print(f"  Saved: {parameters['langmodel']['dawg_path']}")

if __name__ == "__main__":
    train()