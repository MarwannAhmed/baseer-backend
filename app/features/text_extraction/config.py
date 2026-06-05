output_paths = {
    "output": "output",
    "extract": "output/extract",
    "preprocess": "output/preprocess",
    "detect": "output/detect",
    "recognize": "output/recognize",
    "words": "output/words",
    "chars": "output/chars"
}

images = {
    "path": "data/test/sample",
    "format": "jpg"
}

parameters = {
    "extract": {
        "edge_threshold": 125,
        "expand_frac": 0.2
    },
    "orient": {
        "course_step": 5,
        "fine_step": 1,
        "fine_range": 5
    },
    "average": {
        "height_min": 0.3,
        "height_max": 4.5,
        "area_min": 0.05
    },
    "lines": {
        "kde_bandwidth": 0.6,
        "peak_prominence": 0.15,
        "max_assign_dist": 1.2,
        "orphan_min_members": 3,
        "line_pad_fraction": 0.15
    },
    "words": {
        "word_gap_scale": 0.75
    },
    "chars": {
        "min_char_width": 0.1,
        "max_char_width": 1.8,
        "chop_min_valley": 0.35,
        "chop_max_valley": 0.4
    },
    "recognize": {
        "high_conf": 0.75,
        "min_conf": 0.30,
        "min_word_conf": 0.30,
        "max_question_mark_ratio": 0.25,
        "max_candidates": 5,
        "merge_threshold": 0.60
    },
    "features": {
        "norm_size": 64,
        "grid_rows": 8,
        "grid_cols": 8,
        "outline_samples": 64,
        "direction_bins": 16,
        "profile_bins": 16
    },
    "langmodel": {
        "dawg_path": "models/ocr/classical/langmodel/dawg.pkl",
        "bigram_path": "models/ocr/classical/langmodel/bigrams.json",
        "bigram_weight": 0.3,
        "dawg_boost": 0.25,
        "smoothing": 0.01,
        "min_word_frequency": 2
    },
    "SVM": {
        "model_path": "models/svm-text-en/classifier.pkl",
        "scaler_path": "models/svm-text-en/scaler.pkl",
        "encoder_path": "models/svm-text-en/encoder.pkl",
        "TOP_K": 5
    },
    "CNN": {
        "model_path": "models/cnn-ocr/cnn_model.pth",
        "encoder_path": "models/cnn-ocr/cnn_encoder.pkl",
        "TOP_K": 5,
        "norm_size": 28
    },
    "RF": {
        "model_path": "models/ocr/classical/RF/rf_model.pkl",
        "scaler_path": "models/ocr/classical/RF/rf_scaler.pkl",
        "encoder_path": "models/ocr/classical/RF/rf_encoder.pkl",
        "TOP_K": 5
    },
    "train_dirs": {
        "models": "models/ocr/classical",
        "corpus": "data/train/langmodel/books.txt",
        "emnist": "data/train/emnist",
        "chars74k": "data/train/chars74k/EnglishFnt/",
        "synthetic": "data/train/synthetic_chars/"
    }
}

parameters_dl = {
    "unclip_ratio": 1.5,
    "line_thresh": 0.6,
    "det_model_path": "models/pp-ocrv5-mobile-det-infer/",
    "rec_model_path": "models/ar-pp-ocrv5-mobile-rec-infer/"
}