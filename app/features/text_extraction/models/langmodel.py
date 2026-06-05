import json
import pickle
import math
import os

from app.features.text_extraction.config import parameters


class DawgNode:
    def __init__(self):
        self.children = {}
        self.is_end   = False


class LanguageModel:
    def __init__(self):
        self._bigrams = None
        self._dawg    = None

        if os.path.exists(parameters["langmodel"]["bigram_path"]):
            with open(parameters["langmodel"]["bigram_path"], "r", encoding="utf-8") as f:
                self._bigrams = json.load(f)

        if os.path.exists(parameters["langmodel"]["dawg_path"]):
            with open(parameters["langmodel"]["dawg_path"], "rb") as f:
                self._dawg = pickle.load(f)

    @property
    def available(self):
        return self._bigrams is not None

    def _bigram_score(self, sequence):
        if self._bigrams is None or len(sequence) < 2:
            return 0.0
        seq   = sequence.lower()
        score = 0.0
        count = 0
        for a, b in zip(seq, seq[1:]):
            if a in self._bigrams and b in self._bigrams[a]:
                score += math.log(self._bigrams[a][b] + 1e-10)
                count += 1
        return score / count if count > 0 else 0.0

    def _in_dawg(self, word):
        if self._dawg is None:
            return False
        node = self._dawg
        for ch in word.lower():
            if ch not in node.children:
                return False
            node = node.children[ch]
        return node.is_end

    def rescore(self, candidates, context=""):
        if not self.available or not candidates:
            return candidates

        rescored = []
        for char, clf_conf in candidates:
            seq          = (context + char).strip()
            bigram_score = self._bigram_score(seq) if len(seq) >= 2 else 0.0
            bigram_norm  = 1.0 / (1.0 + math.exp(-bigram_score))
            combined     = (1.0 - parameters["langmodel"]["bigram_weight"]) * clf_conf + parameters["langmodel"]["bigram_weight"] * bigram_norm
            rescored.append((char, combined))

        rescored.sort(key=lambda x: x[1], reverse=True)
        return rescored

    def rescore_word(self, word, clf_conf):
        if not self.available:
            return clf_conf
        bigram   = self._bigram_score(word)
        bigram_n = 1.0 / (1.0 + math.exp(-bigram))
        in_dict  = self._in_dawg(word)
        boost    = parameters["langmodel"]["dawg_boost"] if in_dict else 0.0
        return min(1.0, (1.0 - parameters["langmodel"]["bigram_weight"]) * clf_conf + parameters["langmodel"]["bigram_weight"] * bigram_n + boost)