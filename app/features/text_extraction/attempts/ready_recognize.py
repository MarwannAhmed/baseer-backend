import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from paddleocr import TextRecognition

REC_MODEL_DIR = "models/pp-ocrv5_mobile_rec_infer"
MIN_CONFIDENCE = 0.7


def recognize():
    words_dir = "output/words"
    if not os.path.exists(words_dir) or not os.listdir(words_dir):
        print("Run detect.py first (no word crops found)")
        return

    rec_model = TextRecognition(
        model_name="PP-OCRv5_mobile_rec",
        model_dir=REC_MODEL_DIR
    )

    lines = {}
    for fname in sorted(os.listdir(words_dir)):
        if not fname.endswith(".png"):
            continue
        parts = fname.replace(".png", "").split("_")
        li = int(parts[1][1:])
        x  = int(parts[2][1:])
        fpath = os.path.join(words_dir, fname)

        for res in rec_model.predict(input=fpath, batch_size=1):
            rec_dict = res["res"] if "res" in res else res
            text  = rec_dict.get("rec_text", "")
            score = float(rec_dict.get("rec_score", 0))
            if text and score >= MIN_CONFIDENCE:
                lines.setdefault(li, []).append((x, text))

    text_out = "\n".join(
        " ".join(text for _, text in sorted(words))
        for li, words in sorted(lines.items())
    )

    with open("output/recognize.txt", "w", encoding="utf-8") as f:
        f.write(text_out)

    print(f"[3]  Recognition done")
    return text_out