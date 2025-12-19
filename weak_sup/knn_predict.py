from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def knn_predict(Z, idx_all, anchors, k=5, reject_min_prob=0.55):
    idx_to_row = {int(idx): r for r, idx in enumerate(idx_all)}
    Xl, yl = [], []
    for c, idxs in anchors.items():
        for ds_idx in idxs:
            if ds_idx in idx_to_row:
                Xl.append(Z[idx_to_row[ds_idx]])
                yl.append(int(c))
    Xl = np.stack(Xl)
    yl = np.array(yl)

    knn = KNeighborsClassifier(n_neighbors=min(k, len(yl)), weights="distance")
    knn.fit(Xl, yl)

    proba = knn.predict_proba(Z)
    pred = knn.classes_[proba.argmax(axis=1)]
    conf = proba.max(axis=1)

    pred = pred.astype(int)
    pred[conf < reject_min_prob] = -1
    return pred, conf
