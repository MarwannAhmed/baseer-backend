import os
import re
from collections import Counter

GT_DIR      = "data/test/synthetic/ground-truths"
RESULTS_DIR = "data/test/synthetic/dl-results"


def normalize(text):
    text = re.sub(r'[\[\]]',  ' ', text)
    text = re.sub(r'_+',      ' ', text)
    text = re.sub(r'\*+',     ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text):
    return [w for w in text.split() if w]

def char_levenshtein(a, b):
    n = len(b)
    dp = list(range(n + 1))
    for ch_a in a:
        prev, dp[0] = dp[0], dp[0] + 1
        for j, ch_b in enumerate(b, 1):
            prev, dp[j] = dp[j], prev if ch_a == ch_b else 1 + min(prev, dp[j], dp[j - 1])
    return dp[n]

def word_levenshtein(a, b):
    n = len(b)
    dp = list(range(n + 1))
    for tok_a in a:
        prev, dp[0] = dp[0], dp[0] + 1
        for j, tok_b in enumerate(b, 1):
            prev, dp[j] = dp[j], prev if tok_a == tok_b else 1 + min(prev, dp[j], dp[j - 1])
    return dp[n]

def word_recall(gt_tokens, res_tokens):
    if not gt_tokens:
        return 1.0
    gt_counts  = Counter(gt_tokens)
    res_counts = Counter(res_tokens)
    matched    = sum(min(gt_counts[w], res_counts[w]) for w in gt_counts)
    return matched / len(gt_tokens)

def cer(gt_text, res_text):
    if not gt_text:
        return 0.0
    return char_levenshtein(res_text, gt_text) / len(gt_text)

def wer(gt_tokens, res_tokens):
    if not gt_tokens:
        return 0.0
    return word_levenshtein(res_tokens, gt_tokens) / len(gt_tokens)

def evaluate_pair(gt_path, res_path):
    with open(gt_path,  "r", encoding="utf-8") as f:
        gt_norm  = normalize(f.read())
    with open(res_path, "r", encoding="utf-8") as f:
        res_norm = normalize(f.read())

    gt_tokens  = tokenize(gt_norm)
    res_tokens = tokenize(res_norm)

    return {
        "word_recall": word_recall(gt_tokens, res_tokens),
        "cer":         cer(gt_norm, res_norm),
        "wer":         wer(gt_tokens, res_tokens),
        "gt_words":    len(gt_tokens),
        "res_words":   len(res_tokens),
    }

def main():
    gt_files  = set(f for f in os.listdir(GT_DIR)      if f.endswith(".txt"))
    res_files = set(f for f in os.listdir(RESULTS_DIR) if f.endswith(".txt"))
    common    = sorted(gt_files & res_files)

    if not common:
        print("No matching files found.")
        return

    col = {"file": 20, "wr": 13, "cer": 8, "wer": 8, "gt": 10, "res": 10}
    header = (f"{'File':<{col['file']}} {'Word Recall':>{col['wr']}} "
              f"{'CER':>{col['cer']}} {'WER':>{col['wer']}} "
              f"{'GT Words':>{col['gt']}} {'Res Words':>{col['res']}}")
    sep = "-" * len(header)

    print(header)
    print(sep)

    totals = {"word_recall": 0.0, "cer": 0.0, "wer": 0.0}

    for fname in common:
        m = evaluate_pair(
            os.path.join(GT_DIR,      fname),
            os.path.join(RESULTS_DIR, fname),
        )
        for k in totals:
            totals[k] += m[k]

        print(f"{fname:<{col['file']}} {m['word_recall']:>{col['wr']}.1%} "
              f"{m['cer']:>{col['cer']}.1%} {m['wer']:>{col['wer']}.1%} "
              f"{m['gt_words']:>{col['gt']}} {m['res_words']:>{col['res']}}")

    n = len(common)
    print(sep)
    print(f"{'AVERAGE':<{col['file']}} {totals['word_recall']/n:>{col['wr']}.1%} "
          f"{totals['cer']/n:>{col['cer']}.1%} {totals['wer']/n:>{col['wer']}.1%}")
    print(f"\nEvaluated {n} / {len(gt_files)} files  "
          f"({'missing: ' + ', '.join(sorted(gt_files - res_files)) if gt_files - res_files else 'all matched'})")

if __name__ == "__main__":
    main()