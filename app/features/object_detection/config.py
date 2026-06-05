import os

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
MODELS_DIR    = os.path.join(_PROJECT_ROOT, "models/object")

# training
COCO_ROOT       = "coco"
ANNOTATIONS_DIR = os.path.join(COCO_ROOT, "annotations")
IMAGES_DIR      = os.path.join(COCO_ROOT, "images")
ANNOTATION_FILE = os.path.join(ANNOTATIONS_DIR, "instances_train2017.json")
DATA_DIR        = "data"

CLASSES = ["person", "bottle", "chair", "dining table", "handbag"]

WINDOW_SIZE = (96, 96)

HOG_ORIENTATIONS    = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM      = "L2-Hys"

SIFT_GRID_SIZE = 4
SIFT_BINS      = 8

CANNY_GRID_SIZE = 8

MAX_POSITIVES_PER_CLASS = 5000
MAX_NEGATIVES           = 5000
MIN_OBJECT_SIZE         = 32
BBOX_PADDING            = 16

SVM_C           = 0.01
HARD_NEG_MAX    = 1500
HARD_NEG_IMAGES = 150

SCALES                = [1.0, 0.75, 0.5, 0.35, 0.25]
WINDOW_STRIDE         = 16
CONFIDENCE_THRESHOLD  = 0.85
NMS_OVERLAP_THRESHOLD = 0.15
SS_MAX_PROPOSALS      = 500
SS_MIN_BOX_SIZE       = 48