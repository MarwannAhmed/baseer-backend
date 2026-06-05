from app.core.image_utils import decode_image
from app.features.text_extraction.extract.extract import extract
from app.features.text_extraction.preprocess.preprocess import preprocess
from app.features.text_extraction.detect.detect import detect
from app.features.text_extraction.recognize.recognize import recognize


def run_extraction(image_bytes: bytes):
    img = decode_image(image_bytes)
    frames = extract(img)

    all_char_entries  = []
    all_word_line_map = {}

    for i, frame in enumerate(frames):
        binary = preprocess(frame, frame_number=i)
        char_entries, word_line_map = detect(binary, frame_number=i)
        all_char_entries.extend(char_entries)
        all_word_line_map.update(word_line_map)

    return recognize(char_entries=all_char_entries, word_line_map=all_word_line_map)