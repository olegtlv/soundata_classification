import numpy as np

def anchors_from_arrays(idx_all, y_true, n_per_class=1):
    anchors = {}
    for c in np.unique(y_true):
        rows = np.where(y_true == c)[0]
        chosen = rows[:n_per_class]
        anchors[int(c)] = [int(idx_all[r]) for r in chosen]
    return anchors

def compute_prototypes(Z, idx_all, anchors):
    """
    Z: [N, D] embeddings aligned with idx_all
    anchors: {c: [dataset_idx,...]}
    """
    idx_to_row = {int(idx): r for r, idx in enumerate(idx_all)}
    protos = {}
    for c, idxs in anchors.items():
        rows = [idx_to_row[i] for i in idxs if i in idx_to_row]
        if len(rows) == 0:
            continue
        protos[c] = Z[rows].mean(axis=0)
    return protos  # dict class -> [D]


def prototype_predict(Z, protos, reject_margin=0.05):
    """
    returns: pred_class (or -1 for reject), confidence
    confidence = (best_sim - second_best_sim)
    """
    classes = sorted(protos.keys())
    P = np.stack([protos[c] for c in classes], axis=0)  # [C, D]

    sims = Z @ P.T  # [N, C] because Z normalized
    best = sims.argmax(axis=1)
    best_sim = sims[np.arange(len(Z)), best]
    # second best for margin
    sims_sorted = np.sort(sims, axis=1)
    second_sim = sims_sorted[:, -2] if sims.shape[1] >= 2 else -1.0

    margin = best_sim - second_sim
    pred = np.array([classes[i] for i in best], dtype=int)
    pred[margin < reject_margin] = -1

    return pred, margin, best_sim


def expand_with_pseudolabels(idx_all, pred, margin, anchors, add_per_class=5, min_margin=0.15):
    """
    Adds up to add_per_class pseudo labels per class with highest margin.
    """
    used = set(i for lst in anchors.values() for i in lst)

    for c in set(pred):
        if c == -1:
            continue
        # candidates of this class not already used
        cand = [(i, m) for i, (p, m) in enumerate(zip(pred, margin)) if p == c and m >= min_margin]
        # sort by margin descending
        cand.sort(key=lambda t: t[1], reverse=True)
        added = 0
        for row, m in cand:
            ds_idx = int(idx_all[row])
            if ds_idx in used:
                continue
            anchors.setdefault(int(c), [])
            anchors[int(c)].append(ds_idx)
            used.add(ds_idx)
            added += 1
            if added >= add_per_class:
                break
    return anchors
