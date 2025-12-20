from collections import defaultdict
import numpy as np
import random

def indices_by_class(ds, min_conf=None):
    by_c = defaultdict(list)
    for i in range(len(ds)):
        item = ds[i]
        y = int(item["label"])
        conf = float(item.get("confidence", 1.0))
        if (min_conf is None) or (conf >= min_conf):
            by_c[y].append((i, conf))
    # sort high-confidence first (optional)
    for c in by_c:
        by_c[c].sort(key=lambda t: t[1], reverse=True)
    return by_c

import random

def choose_anchors(ds, n_per_class=1, seed=0, min_conf=None, prefer_high_conf=True):
    rng = random.Random(seed)
    by_c = indices_by_class(ds, min_conf=min_conf)

    anchors = {}  # class -> list of indices

    for c, pairs in by_c.items():
        idxs = [i for i, conf in pairs]
        if not prefer_high_conf:
            rng.shuffle(idxs)

        # n_per_class can be: int or dict {class: n}
        n = n_per_class[c] if isinstance(n_per_class, dict) else int(n_per_class)
        anchors[c] = idxs[:n] if n > 0 else []

    return anchors  # {c: [idx,...]}


def add_more_anchors(ds, anchors, add_counts, seed=0, min_conf=None):
    """
    anchors: existing {c: [idx,...]}
    add_counts: dict {c: k_to_add}
    """
    rng = random.Random(seed)
    by_c = indices_by_class(ds, min_conf=min_conf)

    used = set(i for lst in anchors.values() for i in lst)

    for c, k_add in add_counts.items():
        candidates = [i for i, conf in by_c.get(c, []) if i not in used]
        rng.shuffle(candidates)  # or keep sorted by confidence if you prefer
        anchors.setdefault(c, [])
        anchors[c].extend(candidates[:k_add])
        used.update(candidates[:k_add])

    return anchors



def select_n_per_class(ds, n_per_class=3, seed=0, min_conf=None, prefer_high_conf=True):
    """
    Returns a list of dataset indices (ints), selecting up to n_per_class per label.
    Works even if classes are imbalanced.
    """
    rng = random.Random(seed)
    by_c = defaultdict(list)

    for i in range(len(ds)):
        item = ds[i]
        y = int(item["label"])
        conf = float(item.get("confidence", 1.0))
        if (min_conf is None) or (conf >= min_conf):
            by_c[y].append((i, conf))

    # sort by confidence desc if requested
    indices = []
    for c, pairs in by_c.items():
        if prefer_high_conf:
            pairs.sort(key=lambda t: t[1], reverse=True)
            chosen = [i for i, _ in pairs[:n_per_class]]
        else:
            cand = [i for i, _ in pairs]
            rng.shuffle(cand)
            chosen = cand[:n_per_class]
        indices.extend(chosen)

    return indices


def anchors_from_labeled_indices(ds, labeled_indices):
    anchors = defaultdict(list)
    for i in labeled_indices:
        y = int(ds[i]["label"])
        anchors[y].append(int(i))
    return dict(anchors)
