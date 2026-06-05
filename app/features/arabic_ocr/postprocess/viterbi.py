import numpy as np
from arabic_ocr.config import BIGRAM_WEIGHT, MIN_CONF


def viterbi_decode(candidates_per_position, lm):

    if not candidates_per_position:
        return [], 0.0

    T = len(candidates_per_position)

    dp: list[dict[str, tuple[float, str | None]]] = [{}]

    for char, prob in candidates_per_position[0]:
        dp[0][char] = (np.log(max(prob, 1e-9)), None)

    if not dp[0]:
        char, prob = candidates_per_position[0][0]
        dp[0][char] = (np.log(max(prob, 1e-9)), None)

    for t in range(1, T):
        dp.append({})
        for char, prob in candidates_per_position[t]:
            log_emit = np.log(max(prob, 1e-9))
            best_score = -np.inf
            best_previous: str | None = None

            for previous_char, (previous_score, _) in dp[t - 1].items():
                bigram_p = lm.bigrams.get(previous_char, {}).get(char, 1e-6)
                score = previous_score + log_emit + BIGRAM_WEIGHT * np.log(bigram_p)
                if score > best_score:
                    best_score = score
                    best_previous = previous_char

            dp[t][char] = (best_score, best_previous)

        if not dp[t]:
            char, prob = candidates_per_position[t][0]
            dp[t][char] = (np.log(max(prob, 1e-9)), None)
    last       = dp[T - 1]
    best_last  = max(last, key=lambda c: last[c][0])
    final_score = last[best_last][0]

    path = [best_last]
    for t in range(T - 1, 0, -1):
        _, previous = dp[t][path[-1]]
        path.append(previous if previous is not None else "")
    path.reverse()

    return path, float(final_score)
