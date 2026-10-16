"""Unicode-aware, conventional word-level Levenshtein error rate."""
import re
import unicodedata


def tokens(text):
    text = unicodedata.normalize("NFKC", str(text)).casefold().replace("’", "'")
    return re.findall(r"[^\W_]+(?:'[^\W_]+)*", text)


def word_errors(reference, hypothesis):
    a, b = tokens(reference), tokens(hypothesis)
    # Tuple: edit distance, substitutions, deletions, insertions.
    previous = [(j, 0, 0, j) for j in range(len(b) + 1)]
    for i, word in enumerate(a, 1):
        current = [(i, 0, i, 0)]
        for j, other in enumerate(b, 1):
            if word == other:
                current.append(previous[j - 1])
                continue
            d, s, deletion, insertion = previous[j - 1]
            substitute = (d + 1, s + 1, deletion, insertion)
            d, s, deletion, insertion = previous[j]
            delete = (d + 1, s, deletion + 1, insertion)
            d, s, deletion, insertion = current[j - 1]
            insert = (d + 1, s, deletion, insertion + 1)
            current.append(min((substitute, delete, insert), key=lambda x: x[0]))
        previous = current
    d, s, deletion, insertion = previous[-1]
    return {"wer": d / max(1, len(a)), "substitutions": s, "deletions": deletion,
            "insertions": insertion, "reference_words": len(a)}
