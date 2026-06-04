import os
import json
import numpy as np
import cv2
from paddleocr import TextDetection, TextRecognition

from app.features.text_extraction.config import output_paths, parameters_dl


def load_models():
    det = TextDetection(
        model_name="PP-OCRv5_mobile_det",
        model_dir=parameters_dl["det_model_path"],
        unclip_ratio=parameters_dl["unclip_ratio"],
    )
    rec = TextRecognition(
        model_name="PP-OCRv5_mobile_rec",
        model_dir=parameters_dl["rec_model_path"],
    )
    return det, rec

def run_detection(det_model, img):
    raw_output = det_model.predict(img, batch_size=1)
    dt_polys, dt_scores = [], []
    for res in raw_output:
        r = res["res"] if "res" in res else res
        if "dt_polys" in r:
            polys  = r["dt_polys"]
            scores = r.get("dt_scores", [])
            dt_polys  = polys.tolist()  if isinstance(polys,  np.ndarray) else polys
            dt_scores = scores.tolist() if hasattr(scores, "tolist")      else scores
            break
    return raw_output, dt_polys, dt_scores

def crop_polygon(img, polygon):
    if img is None:
        return None
    pts = np.array(polygon, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    return img[y:y + h, x:x + w]

def run_recognition(rec_model, image_path, dt_polys, dt_scores):
    regions = []
    for polygon, det_score in zip(dt_polys, dt_scores):
        crop = crop_polygon(image_path, polygon)
        if crop is None:
            continue
 
        rec_text, rec_score = "", 0.0
        for res in rec_model.predict(input=crop, batch_size=1):
            r = res["res"] if "res" in res else res
            if "rec_text" in r:
                rec_text  = r["rec_text"]
                rec_score = float(r.get("rec_score", 0))
                break
 
        regions.append({
            "polygon":   polygon,
            "det_score": float(det_score),
            "text":      rec_text,
            "rec_score": rec_score,
        })
 
    return regions

def assemble_lines(regions):
    if not regions:
        return []

    def poly_stats(polygon):
        pts = np.array(polygon)
        return pts[:, 0].mean(), pts[:, 1].mean(), pts[:, 1].max() - pts[:, 1].min()

    annotated = [(poly_stats(r["polygon"]), r) for r in regions]
    annotated.sort(key=lambda a: a[0][1])

    avg_h     = np.mean([a[0][2] for a in annotated])
    threshold = avg_h * parameters_dl["line_thresh"]

    lines, current = [], [annotated[0]]
    for item in annotated[1:]:
        line_cy = np.mean([a[0][1] for a in current])
        if abs(item[0][1] - line_cy) <= threshold:
            current.append(item)
        else:
            lines.append(current)
            current = [item]
    lines.append(current)

    return [
        [entry[1] for entry in sorted(line, key=lambda a: a[0][0])]
        for line in lines
    ]

def format_output(lines):
    return "\n".join(
        " ".join(r["text"] for r in line if r["text"])
        for line in lines
    )

def save_results(raw_det_output, regions):
    os.makedirs(output_paths["detect"], exist_ok=True)
    os.makedirs(output_paths["recognize"], exist_ok=True)

    for res in raw_det_output:
        res.save_to_img(save_path=output_paths["detect"])
        res.save_to_json(save_path=output_paths["detect"])

    rec_path = os.path.join(output_paths["recognize"], "recognition_results.json")
    with open(rec_path, "w", encoding="utf-8") as f:
        json.dump(regions, f, indent=2, ensure_ascii=False)

def infer(img, det_model=None, rec_model=None, save_output=False):
    if det_model is None or rec_model is None:
        det_model, rec_model = load_models()

    raw_det_output, dt_polys, dt_scores = run_detection(det_model, img)
    regions = run_recognition(rec_model, img, dt_polys, dt_scores)

    if save_output:
        save_results(raw_det_output, regions)

    lines = assemble_lines(regions)
    return format_output(lines)