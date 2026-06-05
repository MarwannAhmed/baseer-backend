from pathlib import Path
import numpy as np
from itertools import groupby
from arabic_ocr.preprocess import preprocess
from arabic_ocr.segment import segment, CharCrop, segment_lines
from arabic_ocr.classifiers import get_classifier, BaseClassifier
from arabic_ocr.postprocess import ArabicLanguageModel, postprocess
from arabic_ocr.postprocess.reranker import RERANKER
from arabic_ocr.utils.image_io import load_image, resize_if_large
from arabic_ocr.utils.arabic_utils import filter_candidates_by_position, filter_candidates_by_dots
from arabic_ocr.utils import arabic_utils as _au
from arabic_ocr import config as cfg
from arabic_ocr.utils.visualize import draw_lines, draw_paws, draw_chars, draw_dots, save_debug_visualization
from arabic_ocr.config import OUTPUT_DIR
import logging

logger = logging.getLogger(__name__)


class ArabicOCRPipeline:

    def __init__(self, classifier = "svm", debug = False):
        self.classifier: BaseClassifier = get_classifier(classifier)
        self.lm = ArabicLanguageModel()
        self.debug = debug

        if debug:
            import arabic_ocr.config as cfg
            cfg.DEBUG = True

    def run(self, image_path, frame_number = 0):
        img = load_image(image_path)
        return self._process(img, frame_number=frame_number)

    def run_array(self, img, frame_number = 0):
        return self._process(img, frame_number=frame_number)

    def run_batch(self, image_paths):
        return [self.run(p, frame_number=i) for i, p in enumerate(image_paths)]


    def _process(self, img: np.ndarray, frame_number: int = 0) -> str:
        img = resize_if_large(img)

        binary = preprocess(img, frame_number=frame_number)

        char_crops: list[CharCrop] = segment(binary)

        if not char_crops:
            return ""

        if self.debug:
            self._save_debug(binary, char_crops, frame_number)

        imgs      = [c.img  for c in char_crops]
        dot_lists = [c.dots for c in char_crops]
        try:
            all_candidates = self.classifier.predict_batch(imgs, dot_lists)
        except Exception:
            logger.exception("Classifier.predict_batch failed — falling back to undetected candidates")
            all_candidates = [[("", 1.0)] for _ in imgs]

        if len(all_candidates) != len(char_crops):
            logger.warning("Classifier returned %d candidate lists for %d crops; normalising.",
                           len(all_candidates), len(char_crops))
            all_candidates = list(all_candidates[:len(char_crops)])
            while len(all_candidates) < len(char_crops):
                all_candidates.append([("", 1.0)])

        from arabic_ocr.features.dot_features import dot_features
        for crop, cands in zip(char_crops, all_candidates):
            df = dot_features(crop.dots)
            crop.candidates = RERANKER.rerank(cands, df.tolist())

        return postprocess(char_crops, self.lm)

    def _save_debug(self, binary, crops, frame):
        debug_dir = OUTPUT_DIR / "debug" / f"{frame:04d}"

        lines = segment_lines(binary)
        line_bounds = [(y1, y2) for y1, y2, _ in lines]

        # Individual character boxes
        char_boxes = [
            (c.abs_x, c.abs_y,
             c.abs_x + c.img.shape[1], c.abs_y + c.img.shape[0])
            for c in crops
        ]

        # PAW boxes: union of all char boxes sharing (line_idx, paw_idx)
        paw_boxes: list[tuple[int, int, int, int]] = []
        for _, group in groupby(crops, key=lambda c: (c.line_idx, c.paw_idx)):
            grp = list(group)
            x1 = min(c.abs_x for c in grp)
            y1 = min(c.abs_y for c in grp)
            x2 = max(c.abs_x + c.img.shape[1] for c in grp)
            y2 = max(c.abs_y + c.img.shape[0] for c in grp)
            paw_boxes.append((x1, y1, x2, y2))

        all_dots = [d for c in crops for d in c.dots]

        save_debug_visualization(draw_lines(binary, line_bounds),"lines", debug_dir)
        save_debug_visualization(draw_paws(binary, paw_boxes),"paws", debug_dir)
        save_debug_visualization(draw_chars(binary, char_boxes),"chars", debug_dir)
        save_debug_visualization(draw_dots(binary, all_dots),"dots", debug_dir)


def _filter_candidates_with_fallback(candidates, position, dots_above, dots_below):
    filtered = filter_candidates_by_position(candidates, position)

    observed = (dots_above, dots_below)
    boost = getattr(cfg, "DOT_RERANK_BOOST", 0.20)
    penalty = getattr(cfg, "DOT_RERANK_PENALTY", 0.10)

    def expected_for_label(label: str):
        base = label.rsplit("_", 1)[0]
        return _au._LETTER_DOT_COUNTS.get(base)

    reranked = []
    for label, conf in filtered:
        expected = expected_for_label(label)
        new_conf = conf
        if expected is not None:
            if expected == observed:
                new_conf = conf + boost
            else:
                # if we observed no dots but candidate expects dots, penalise
                new_conf = max(0.0, conf - penalty)
        reranked.append((label, new_conf))

    # Keep original ordering for equal scores; sort by adjusted confidence
    reranked.sort(key=lambda t: t[1], reverse=True)

    if not reranked:
        return candidates
    total = sum(c for _, c in reranked) or 1.0
    normalized = [(lab, c / total) for lab, c in reranked]
    return normalized
