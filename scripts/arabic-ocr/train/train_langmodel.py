import argparse
import bz2
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path
from app.features.arabic_ocr.config import DATA_DIR, MODELS_DIR
from app.features.arabic_ocr.postprocess.dawg import build_dawg, save_dawg

# Only Arabic script characters; excludes punctuation, digits, Latin.
_ARABIC_RE = re.compile(r"[؀-ۿ]+")


MAX_DAWG_WORDS = 500_000


def extract_arabic_tokens(dump_path):
    skip_re = re.compile(r"^\s*[|{}\[\]<#*]")  

    with bz2.open(str(dump_path), "rt", encoding="utf-8", errors="replace") as f:
        inside_text = False
        for line in f:
            if "<text" in line:
                inside_text = True
            if inside_text and not skip_re.match(line):
                yield from _ARABIC_RE.findall(line)
            if "</text>" in line:
                inside_text = False


def extract_arabic_tokens_from_plaintext(corpus_path):
    with open(str(corpus_path), encoding="utf-8", errors="replace") as f:
        for line in f:
            yield from _ARABIC_RE.findall(line)


def build_bigrams_from_tokens(token_iter, max_dawg_words = MAX_DAWG_WORDS, report_everyt = 1_000_000):
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    unique_words: set[str] = set()
    total_tokens = 0

    for word in token_iter:
        if len(word) < 2:
            continue
        # Character bigrams within the word
        for a, b in zip(word, word[1:]):
            counts[a][b] += 1
        # Collect word for DAWG (cap to avoid unbounded memory)
        if len(unique_words) < max_dawg_words:
            unique_words.add(word)
        total_tokens += 1
        # if total_tokens % report_every == 0:
        #     print(f"  … {total_tokens:,} tokens processed, "
        #           f"{len(unique_words):,} unique words")

    print(f"Total tokens: {total_tokens:,}   Unique words: {len(unique_words):,}")

    # Normalise to probabilities
    bigrams: dict[str, dict[str, float]] = {}
    for a, nexts in counts.items():
        total = sum(nexts.values())
        bigrams[a] = {b: cnt / total for b, cnt in nexts.items()}

    return bigrams, unique_words


def main():
    parser = argparse.ArgumentParser(
        description="Train Arabic bigram LM + DAWG from Wikipedia dump or plain corpus"
    )
    parser.add_argument(
        "--dump",
        default=str(DATA_DIR / "corpus" / "arwiki-latest-pages-articles.xml.bz2"),
        help="Path to arwiki …xml.bz2 dump (preferred source)",
    )
    parser.add_argument(
        "--corpus",
        default=str(DATA_DIR / "corpus" / "arabic_corpus.txt"),
        help="Plain-text Arabic corpus (fallback when --dump not found)",
    )
    parser.add_argument(
        "--wordlist",
        default=str(DATA_DIR / "corpus" / "arabic_wordlist.txt"),
        help="Optional extra word list to supplement the DAWG",
    )
    parser.add_argument(
        "--out-dir",
        default=str(MODELS_DIR / "langmodel-ocr-ar"),
        help="Output directory for bigrams.json and dawg.pkl",
    )
    parser.add_argument(
        "--max-dawg-words", type=int, default=MAX_DAWG_WORDS,
        help=f"Max unique words stored in the DAWG (default {MAX_DAWG_WORDS:,})",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dump_path    = Path(args.dump)
    corpus_path  = Path(args.corpus)

    if dump_path.exists():
        print(f"Streaming Wikipedia dump: {dump_path}")
        token_iter = extract_arabic_tokens(dump_path)
    elif corpus_path.exists():
        print(f"Dump not found — falling back to plain corpus: {corpus_path}")
        token_iter = extract_arabic_tokens_from_plaintext(corpus_path)
    else:
        print("No corpus source found. Provide one of:")
        print(f"  --dump    {dump_path}")
        print(f"  --corpus  {corpus_path}")
        return

    print("Building bigrams and collecting vocabulary …")
    bigrams, unique_words = build_bigrams_from_tokens(
        token_iter, max_dawg_words=args.max_dawg_words
    )

    wordlist_path = Path(args.wordlist)
    if wordlist_path.exists():
        extra = {
            w.strip() for w in wordlist_path.read_text(encoding="utf-8").splitlines()
            if w.strip()
        }
        before = len(unique_words)
        unique_words.update(extra)
        print(f"Added {len(unique_words) - before:,} words from word list")

    bigrams_out = out_dir / "bigrams.json"
    with open(bigrams_out, "w", encoding="utf-8") as f:
        json.dump(bigrams, f, ensure_ascii=False)
    print(f"Bigrams saved  → {bigrams_out}  ({len(bigrams):,} initial chars)")

    print(f"Building DAWG trie for {len(unique_words):,} words …")
    dawg_root = build_dawg(list(unique_words))
    dawg_out  = out_dir / "dawg.pkl"
    save_dawg(dawg_root, dawg_out)
    print(f"DAWG saved     → {dawg_out}")

    print("\nDone. Run the OCR pipeline — it will auto-load models from:")
    print(f"  {out_dir}")


if __name__ == "__main__":
    main()
