import os
import pickle

import cv2
from collections import defaultdict
import numpy as np
from textblob import TextBlob

from app.features.text_extraction.config import output_paths, parameters
from app.features.text_extraction.models.svm_classifier import SVMClassifier
# from app.features.text_extraction.models.cnn_classifier import CNNClassifier
# from app.features.text_extraction.models.rf_classifier  import RFClassifier


def _load_dawg():
    if not os.path.exists(parameters["langmodel"]["dawg_path"]):
        print("Warning: DAWG not found, falling back to top-1 predictions")
        return None
    with open(parameters["langmodel"]["dawg_path"], "rb") as f:
        return pickle.load(f)

def _in_dawg(dawg, word):
    if dawg is None:
        return False
    if word and all(c.isdigit() for c in word):
        return True
    node = dawg
    for ch in word.lower():
        if ch not in node.children:
            return False
        node = node.children[ch]
    return node.is_end

def _is_valid_prefix(dawg, prefix):
    if dawg is None:
        return True
    if prefix and all(c.isdigit() for c in prefix):
        return True
    node = dawg
    for ch in prefix.lower():
        if ch not in node.children:
            return False
        node = node.children[ch]
    return True

def _search_all_combinations(dawg, candidates_per_pos, locked_mask, current, pos, all_matches, current_confidences, word_id=None, depth=0):
    if pos == len(candidates_per_pos):
        if _in_dawg(dawg, current):
            all_matches.append((current, current_confidences.copy()))
        return
    
    if locked_mask[pos]:
        char, conf = candidates_per_pos[pos][0]
        if _is_valid_prefix(dawg, current + char):
            current_confidences.append(conf)
            _search_all_combinations(dawg, candidates_per_pos, locked_mask, 
                                     current + char, pos + 1, all_matches, current_confidences, word_id, depth + 1)
            current_confidences.pop()
    else:
        for idx, (char, conf) in enumerate(candidates_per_pos[pos]):
            if _is_valid_prefix(dawg, current + char):
                current_confidences.append(conf)
                _search_all_combinations(dawg, candidates_per_pos, locked_mask,
                                        current + char, pos + 1, all_matches, current_confidences,
                                        word_id, depth + 1)
                current_confidences.pop()


def _search_all_combinations_with_merge(dawg, augmented_candidates, locked_mask, current, pos, all_matches, current_confidences, word_id=None, depth=0):
    if pos == len(augmented_candidates):
        if _in_dawg(dawg, current):
            all_matches.append((current, current_confidences.copy()))
        return
    
    if locked_mask[pos]:
        char, conf = augmented_candidates[pos][0]
        if _is_valid_prefix(dawg, current + char):
            current_confidences.append(conf)
            _search_all_combinations_with_merge(dawg, augmented_candidates, locked_mask,
                                                current + char, pos + 1, all_matches, current_confidences,
                                                word_id, depth + 1)
            current_confidences.pop()
    else:
        for idx, (char, conf) in enumerate(augmented_candidates[pos]):
            if _is_valid_prefix(dawg, current + char):
                current_confidences.append(conf)
                _search_all_combinations_with_merge(dawg, augmented_candidates, locked_mask,
                                                    current + char, pos + 1, all_matches, current_confidences,
                                                    word_id, depth + 1)
                current_confidences.pop()


def _try_merge_characters(clf, img1, img2, pos=None):
    if img1 is None or img2 is None:
        return None, 0.0
    
    h = max(img1.shape[0], img2.shape[0])
    img1_resized = cv2.resize(img1, (img1.shape[1], h))
    img2_resized = cv2.resize(img2, (img2.shape[1], h))
    
    merged_img = np.hstack([img1_resized, img2_resized])
    
    candidates = clf.predict(merged_img)
    merged_char, merged_conf = candidates[0] if candidates else (None, 0.0)
    
    return merged_char, merged_conf


def decode_word_with_search(char_entries, char_images, candidates_list, clf, dawg, word_id):
    if not char_entries:
        return "", 0.0
    
    n_chars = len(char_entries)
    
    locked_mask = []
    candidates_per_pos = []
    top_chars = []
    low_conf_chars = []
    
    for i, candidates in enumerate(candidates_list):
        top_char, top_conf = candidates[0]
        top_chars.append(top_char)
        is_locked = top_conf >= parameters["recognize"]["high_conf"]
        locked_mask.append(is_locked)
        
        if not is_locked:
            low_conf_chars.append(i)
        
        candidates_per_pos.append(candidates)
    
    best_word = None
    best_confidence = -1.0
    all_matches = []
    
    if dawg is not None:
        
        _search_all_combinations(dawg, candidates_per_pos, locked_mask, "", 0, all_matches, [], word_id)
        
        if all_matches:
            for match_word, char_confs in all_matches:
                avg_conf = sum(char_confs) / len(char_confs) if char_confs else 0.0
                
                if avg_conf > best_confidence:
                    best_confidence = avg_conf
                    best_word = match_word
            
            decoded = best_word
            
            if len(decoded) == len(top_chars):
                case_preserved = []
                for j, orig_char in enumerate(top_chars):
                    if orig_char.isupper():
                        case_preserved.append(decoded[j].upper())
                    else:
                        case_preserved.append(decoded[j].lower())
                final_word = "".join(case_preserved)
                return final_word, best_confidence
            else:
                return decoded, best_confidence
    
    low_conf_indices = [i for i, locked in enumerate(locked_mask) if not locked]
    
    if len(low_conf_indices) >= 2 and dawg is not None and not all_matches:
        best_merge_word = None
        best_merge_confidence = -1.0
        all_merge_matches = []
        
        for k in range(len(low_conf_indices) - 1):
            i = low_conf_indices[k]
            j = low_conf_indices[k + 1]
            
            if j != i + 1:
                continue
            
            merged_char, merged_conf = _try_merge_characters(
                clf, char_images[i], char_images[j], i
            )
            
            if merged_char and merged_conf >= parameters["recognize"]["merge_threshold"]:                
                temp_candidates = [list(c) for c in candidates_per_pos]
                temp_candidates[i].append((merged_char + "__MERGE__", merged_conf))
                
                merge_matches = []
                
                _search_all_combinations_with_merge(dawg, temp_candidates, locked_mask, "", 0, 
                                                    merge_matches, [], word_id)
                
                for match_word, char_confs in merge_matches:
                    avg_conf = sum(char_confs) / len(char_confs) if char_confs else 0.0
                    
                    if avg_conf > best_merge_confidence:
                        best_merge_confidence = avg_conf
                        best_merge_word = match_word
                        all_merge_matches = merge_matches
        
        if best_merge_word:
            word = best_merge_word.replace("__MERGE__", "")            
            if len(word) <= len(top_chars):
                case_preserved = []
                for jdx in range(len(word)):
                    if jdx < len(top_chars) and top_chars[jdx].isupper():
                        case_preserved.append(word[jdx].upper())
                    else:
                        case_preserved.append(word[jdx].lower())
                final_word = "".join(case_preserved)
                return final_word, best_merge_confidence
    
    result_chars = []
    fallback_confidences = []
    
    for i, (candidates, locked) in enumerate(zip(candidates_list, locked_mask)):
        top_char, top_conf = candidates[0]
        if not locked and top_conf < parameters['recognize']['min_conf']:
            # result_chars.append("?")
            result_chars.append(top_char)
            fallback_confidences.append(top_conf)
        else:
            result_chars.append(top_char)
            fallback_confidences.append(top_conf)
    
    final_word = "".join(result_chars)
    fallback_confidence = sum(fallback_confidences) / len(fallback_confidences) if fallback_confidences else 0.0
    return final_word, fallback_confidence

def should_filter_word(word, word_confidence):
    if not word:
        return True
    if word_confidence < parameters["recognize"]["min_word_conf"]:
        return True
    question_mark_count = word.count('?')
    question_mark_ratio = question_mark_count / len(word) if len(word) > 0 else 1.0
    if question_mark_ratio >= parameters["recognize"]["max_question_mark_ratio"]:
        return True
    return False

def _load_char_crops():
    entries = []
    for fname in sorted(os.listdir(output_paths["chars"])):
        if not fname.endswith(".png"):
            continue
        parts = fname.replace(".png", "").split("_")
        frame_idx = int(parts[0][1:])
        word_idx = int(parts[1][1:])
        char_idx = int(parts[2][1:])
        x = int(parts[3][1:])
        entries.append((char_idx, frame_idx, word_idx, x, os.path.join(output_paths["chars"], fname)))
    return sorted(entries, key=lambda e: (e[1], e[2], e[3]))

def _load_word_line_map():
    mapping = {}
    for fname in sorted(os.listdir(output_paths["words"])):
        if not fname.endswith(".png"):
            continue
        parts = fname.replace(".png", "").split("_")
        frame_idx = int(parts[0][1:])
        li = int(parts[1][1:])
        word_idx = int(parts[2][1:])
        x = int(parts[3][1:])
        mapping[(frame_idx, word_idx)] = (li, x, frame_idx)
    return mapping

def recognize(char_entries=None, word_line_map=None, save_output=False):
    if char_entries is None:
        if not os.path.exists(output_paths["chars"]) or not os.listdir(output_paths["chars"]):
            print("No character crops found. Run detect first.")
            return ""
        raw_entries = _load_char_crops()
        if not raw_entries:
            print("No character entries found.")
            return ""
        char_entries = [
            (ci, fi, wi, x, cv2.imread(fp, cv2.IMREAD_GRAYSCALE))
            for ci, fi, wi, x, fp in raw_entries
        ]
        word_line_map = _load_word_line_map()

    if not char_entries:
        print("No character entries found.")
        return ""

    clf = SVMClassifier()
    dawg = _load_dawg()

    word_entries = defaultdict(list)
    for ci, fi, wi, x, img in char_entries:
        word_entries[(fi, wi)].append((ci, x, img))

    for key in word_entries:
        word_entries[key].sort(key=lambda e: e[1])

    word_strings = {}
    
    for (frame_idx, word_idx), char_entries_list in word_entries.items():
        images = []
        valid_entries = []
        
        for e in char_entries_list:
            char_idx, x, img = e
            if img is not None and img.size > 0:
                images.append(img)
                valid_entries.append(e)
        
        if not images:
            word_strings[(frame_idx, word_idx)] = ""
            continue
        
        results = clf.predict_batch(images)
        top1_word = "".join(r[0][0] for r in results)
        
        decoded_word, word_confidence = decode_word_with_search(
            valid_entries, images, results, clf, dawg, (frame_idx, word_idx)
        )

        word_strings[(frame_idx, word_idx)] = "___" if should_filter_word(decoded_word, word_confidence) else decoded_word
    
    line_words = {}
    for (frame_idx, word_idx), text in word_strings.items():
        if (frame_idx, word_idx) not in word_line_map:
            print(f"Warning: No line mapping for word {(frame_idx, word_idx)}")
            continue
        li, x, frame_idx = word_line_map[(frame_idx, word_idx)]
        line_words.setdefault((frame_idx, li), []).append((x, text))
    
    raw_text_out = "\n".join(
        " ".join(w for _, w in sorted(line_words[k]))
        for k in sorted(line_words)
    )
    
    if save_output:
        with open(f"{output_paths['recognize']}/recognized.txt", "w", encoding="utf-8") as f:
            f.write(raw_text_out)
    
    corrected_lines = []
    raw_lines = raw_text_out.split('\n')
    
    for line in raw_lines:
        if line.strip():
            line_no_question_marks = line.replace('?', '')
            line_no_underscores = line_no_question_marks.replace('_', '')
            line_lower = line_no_underscores.lower()
            
            blob = TextBlob(line_lower)
            corrected_line = str(blob.correct())
            
            corrected_lines.append(corrected_line)
        else:
            corrected_lines.append(line)
    
    text_out = '\n'.join(corrected_lines)
    
    if save_output:
        with open(f"{output_paths['recognize']}/corrected_recognized.txt", "w", encoding="utf-8") as f:
            f.write(text_out)
            
    return text_out