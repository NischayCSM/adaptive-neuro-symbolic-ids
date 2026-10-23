"""
Vectorised neuro-symbolic fusion variants.

Inputs (N rows, K classes in label_classes.json order):
    P  detector probabilities                       (N x K)
    S  max stored precision of fired rules per class, 0 if none fired (N x K)
    F  fired mask: at least one rule for class k fired              (N x K, bool)
    M  number of fired rules per class                              (N x K, int)
S / F / M come from the KB's ACTIVE rules with the same matching as SymbolicReasoner.predict
(cl_eval.symbolic_results): a rule fires when every condition holds (reasoner.evaluate_condition),
and never when a condition's feature is not in feature_names.

Variants (each returns predicted class indices):
    V0 linear              argmax  alpha * P + (1 - alpha) * S        (= NeuroSymbolicFuser)
    V1 gated               argmax P where max(P) >= tau, V0 elsewhere
    V2 neutral-absence     V0 with S_k replaced by P_k wherever no rule for class k fired
    V3 gated + neutral     V1 gating on top of V2
    V4 stacking            multinomial logistic regression on [log(P + 1e-9), S, F, log1p(M)],
                           class_weight="balanced" (features standardised before the fit)

V0 replicates NeuroSymbolicFuser's arithmetic exactly: alpha * P is computed in P's dtype and
(1 - alpha) * S in float64 then cast to P's dtype before the sum (NumPy 2 scalar promotion).
Exact ties in the top score are broken by argmax (lowest class index). NeuroSymbolicFuser breaks
them by Python set order instead, which depends on PYTHONHASHSEED; top_tied() flags such rows.
"""

import hashlib
from pathlib import Path

import numpy as np

from src.reasoning.reasoner import evaluate_condition


def rule_evidence(kb_path, feature_names, X, class_to_idx):
    """Return (S, F, M) for the ACTIVE rules of the KB at kb_path."""
    from src.rules.knowledge_base import KnowledgeBase

    rules = KnowledgeBase(db_path=kb_path).get_active_rules()
    col = {name: i for i, name in enumerate(feature_names)}  # last index wins, as in the reasoner's dict
    N, K = len(X), len(class_to_idx)
    S = np.zeros((N, K), dtype=np.float64)
    M = np.zeros((N, K), dtype=np.int32)
    for rule in rules:
        mask = np.ones(N, dtype=bool)
        for cond in rule["conditions"]:
            if cond["feature"] not in col:
                mask[:] = False
                break
            mask &= np.asarray(evaluate_condition(X[:, col[cond["feature"]]], cond["operator"], cond["threshold"]),
                               dtype=bool)
        k = class_to_idx[rule["predicted_class"]]
        M[mask, k] += 1
        S[mask, k] = np.maximum(S[mask, k], float(rule.get("precision", 0.0)))
    return S, M > 0, M


def cached_rule_evidence(kb_path, feature_names, X, class_to_idx, cache_path):
    """rule_evidence() cached as .npz; recomputed if the KB file, X or the feature/class lists change."""
    kb_path, cache_path = Path(kb_path), Path(cache_path)
    h = hashlib.sha1()
    h.update(kb_path.read_bytes())
    h.update(np.ascontiguousarray(X).tobytes())
    h.update("|".join(feature_names).encode())
    h.update("|".join(sorted(class_to_idx, key=class_to_idx.get)).encode())
    key = h.hexdigest()
    if cache_path.exists():
        z = np.load(cache_path)
        if str(z["key"]) == key:
            return z["S"], z["F"], z["M"]
    S, F, M = rule_evidence(kb_path, feature_names, X, class_to_idx)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, S=S, F=F, M=M, key=key)
    return S, F, M


def linear_scores(P, S, alpha):
    dt = P.dtype.type
    return dt(alpha) * P + ((1 - alpha) * S).astype(P.dtype)


def top_tied(scores):
    """Rows whose top score is shared exactly by two or more classes."""
    return (scores == scores.max(1, keepdims=True)).sum(1) >= 2


def neutral_scores(P, S, F, alpha):
    return linear_scores(P, np.where(F, S, P), alpha)


def v0_linear(P, S, alpha):
    return linear_scores(P, S, alpha).argmax(1)


def v1_gated(P, S, alpha, tau):
    return np.where(P.max(1) >= tau, P.argmax(1), v0_linear(P, S, alpha))


def v2_neutral(P, S, F, alpha):
    return neutral_scores(P, S, F, alpha).argmax(1)


def v3_gated_neutral(P, S, F, alpha, tau):
    return np.where(P.max(1) >= tau, P.argmax(1), v2_neutral(P, S, F, alpha))


def stacking_features(P, S, F, M):
    return np.hstack([np.log(P.astype(np.float64) + 1e-9), S, F.astype(np.float64), np.log1p(M)])


class StackingFuser:
    """V4: multinomial logistic regression over detector and rule evidence.
    detector_only=True uses only the log(P + 1e-9) block (control: recalibration without rules)."""

    def __init__(self, seed=42, max_iter=2000, detector_only=False):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.detector_only = detector_only
        self.model = make_pipeline(StandardScaler(),
                                   LogisticRegression(class_weight="balanced", max_iter=max_iter, random_state=seed))

    def features(self, P, S, F, M):
        X = stacking_features(P, S, F, M)
        return X[:, :P.shape[1]] if self.detector_only else X

    def fit(self, P, S, F, M, y):
        self.model.fit(self.features(P, S, F, M), y)
        return self

    def predict(self, P, S, F, M):
        return self.model.predict(self.features(P, S, F, M))

    def decision_function(self, P, S, F, M):
        return self.model.decision_function(self.features(P, S, F, M))
