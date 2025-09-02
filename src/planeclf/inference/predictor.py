from pathlib import Path
from typing import List, Sequence, Tuple
import numpy as np
from typing import List, Tuple, Sequence, Union


def predict_top1(model, X: np.ndarray, classes: Sequence[str] | None = None
) -> Tuple[Union[np.ndarray, List[str]], np.ndarray]:
    """
    Retourne (labels_predits, probas_top1) pour chaque échantillon.
    - labels_predits : liste de labels si `classes` est fourni, sinon indices np.ndarray
    - probas_top1 : np.ndarray de shape (N,)
    """
    probs = model.predict_proba(X)       
    idx = probs.argmax(axis=1)                  # (N,)
    conf = probs[np.arange(len(idx)), idx]      # (N,)
    if classes is not None:
        labels = [classes[i] for i in idx]
    else:
        labels = idx
    return labels, conf

def predict_topk(model, X: np.ndarray, classes: Sequence[str] | None = None, k: int = 3
) -> List[List[Tuple[Union[str, int], float]]]:
    """
    Retourne pour chaque échantillon une liste de k (label, proba) triés décroissants.
    """
    probs = model.predict_proba(X)              # (N, C)
    k = min(k, probs.shape[1])
    # indices des k meilleures classes (non triés)
    topk_idx = np.argpartition(-probs, k-1, axis=1)[:, :k]
    out = []
    for n in range(probs.shape[0]):
        idxs = topk_idx[n]
        # trier localement par proba décroissante
        order = np.argsort(-probs[n, idxs])
        idxs = idxs[order]
        pairs = []
        for i in idxs:
            label = classes[i] if classes is not None else int(i)
            pairs.append((label, float(probs[n, i])))
        out.append(pairs)
    return out
