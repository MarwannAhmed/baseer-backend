import unicodedata
from typing import TYPE_CHECKING

from .language_model import ArabicLanguageModel
from .viterbi import viterbi_decode
from .beam_search import beam_search_decode
from .dawg import DawgNode, build_dawg, dawg_search, save_dawg, load_dawg
from arabic_ocr.utils.arabic_utils import hmdb_label_to_unicode
from arabic_ocr.config import TOP_K, PAW_SPACE_PENALTY

if TYPE_CHECKING:
    from arabic_ocr.segment import CharCrop

_TATWEEL = "ـ"


_UNDETECTED = "_"


def _to_unicode(label):
    if not label:
        return _UNDETECTED

    if "_" not in label and all("؀" <= ch <= "ۿ" for ch in label):
        return label.replace(_TATWEEL, "")

    converted = hmdb_label_to_unicode(label)
    if converted:
        return converted.replace(_TATWEEL, "")
    return _UNDETECTED


def _to_unicode_candidates(raw_cands):
    merged: dict[str, float] = {}
    for label, conf in raw_cands:
        uc = _to_unicode(label)
        if uc == _UNDETECTED:
            continue
        merged[uc] = merged.get(uc, 0.0) + conf
    if not merged:
        return [(_UNDETECTED, 1.0)]
    total = sum(merged.values())
    if total <= 0.0:
        return [(_UNDETECTED, 1.0)]
    return sorted(
        [(ch, p / total) for ch, p in merged.items()],
        key=lambda t: t[1],
        reverse=True,
    )


def postprocess(char_crops, lm):
    if not char_crops:
        return ""

    from itertools import groupby

    # groups by line_idx to preserve physical line breaks
    lines_text: list[str] = []
    for _, line_group in groupby(char_crops, key=lambda c: c.line_idx):
        line_crops = list(line_group)
        
        # groups raw crops by PAW index
        paws = []
        for _, paw_group in groupby(line_crops, key=lambda c: c.paw_idx):
            paws.append(list(paw_group))
        
        n_paws = len(paws)
        dp = [(0.0, [])] * (n_paws + 1)
        
        for i in range(1, n_paws + 1):
            best_score = -1e99
            best_words = []
            
            max_paws_per_word = 8
            for j in range(max(0, i - max_paws_per_word), i):
                group_crops = []
                for k in range(j, i):
                    group_crops.extend(paws[k])
                    
                # Convert to Unicode before decoding
                candidates_per_pos = [
                    _to_unicode_candidates(getattr(c, "candidates", [("", 1.0)]))
                    for c in group_crops]
 
                beams = beam_search_decode(candidates_per_pos, lm, beam_width=50, return_beams=True)
                
                group_best_score = -1e99
                group_best_word = ""
                for score, path in beams:
                    word_candidate = "".join(ch for ch in path if ch != _UNDETECTED)
                    bonus = lm.rescore_word(word_candidate, clf_conf=0.0) if lm is not None else 0.0
                    final = float(score) + float(bonus)
                    if final > group_best_score:
                        group_best_score = final
                        group_best_word = word_candidate
                
                prev_score, prev_words = dp[j]
                
                penalty = PAW_SPACE_PENALTY if j > 0 else 0.0
                
                total_score = prev_score + group_best_score + penalty
                if total_score > best_score:
                    best_score = total_score
                    # Only append if non-empty
                    if group_best_word.strip():
                        best_words = prev_words + [group_best_word]
                    else:
                        best_words = list(prev_words)
                        
            dp[i] = (best_score, best_words)

        if dp[n_paws][1]:
            lines_text.append(" ".join(dp[n_paws][1]))

    text = "\n".join(lines_text)
    return unicodedata.normalize("NFC", text)


__all__ = [
    "ArabicLanguageModel",
    "viterbi_decode",
    "beam_search_decode",
    "postprocess",
    "DawgNode", "build_dawg", "dawg_search", "save_dawg", "load_dawg",
]
