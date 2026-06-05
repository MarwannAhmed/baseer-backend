import argparse
import cv2
import os
import shutil

from app.features.text_extraction.config import images, output_paths


def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_name', help='Name or path of the image file')
    parser.add_argument('--save-output', action='store_true', help='Store output files')
    args = parser.parse_args()
    
    for path_key in ["output", "preprocess", "detect", "extract", "recognize", "words", "chars"]:
        path = output_paths[path_key]
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    image_name = args.image_name
    if not image_name.endswith(f".{images['format']}"):
        image_path = f"{images['path']}/{image_name}.{images['format']}"
    else:
        image_path = f"{images['path']}/{image_name}"

    assert image_path is not None, "No image path entered"
    
    img = cv2.imread(image_path)
    if args.save_output:
        os.makedirs(output_paths["output"], exist_ok=True)
        cv2.imwrite(f"{output_paths['output']}/original.png", img)

    return img, args.save_output