import cv2
from app.features.object_detection import config


def get_proposals(image):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    raw = ss.process()

    proposals = []
    for (x, y, w, h) in raw[:config.SS_MAX_PROPOSALS]:
        if w < config.SS_MIN_BOX_SIZE or h < config.SS_MIN_BOX_SIZE:
            continue
        proposals.append([x, y, x + w, y + h])

    print(f"  Selective Search: {len(raw)} raw proposals → "
          f"{len(proposals)} after size filter")

    return proposals