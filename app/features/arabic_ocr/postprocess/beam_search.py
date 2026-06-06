import numpy as np

from arabic_ocr.config import BIGRAM_WEIGHT


def beam_search_decode(candidates_per_position, lm, beam_width = 5, return_beams = False):

    if not candidates_per_position:
        return []

    # Each beam: (accumulated_log_score, [chars_so_far])
    beams = [(0.0, [])]

    for candidates in candidates_per_position:
        new_beams: list[tuple[float, list[str]]] = []

        for acc_score, path in beams:
            for char, prob in candidates:
                log_emit = np.log(max(prob, 1e-9))

                if path and lm is not None:
                    prev = path[-1]
                    bigram_p = lm.bigrams.get(prev, {}).get(char, 1e-6)
                    lm_score = np.log(bigram_p) * BIGRAM_WEIGHT
                else:
                    lm_score = 0.0

                new_score = acc_score + log_emit + lm_score
                new_beams.append((new_score, path + [char]))

        # Prune to top beam_width
        new_beams.sort(key=lambda t: t[0], reverse=True)
        beams = new_beams[:beam_width]

    if not beams:
        return [] if not return_beams else []
    # beams are sorted by score desc
    if return_beams:
        return beams
    return beams[0][1]
