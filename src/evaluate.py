import numpy as np
from sklearn.metrics import auc, fbeta_score, precision_recall_curve, precision_score, recall_score


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    probs = model.predict_proba(X_test)[:, 1]

    p, r, t = precision_recall_curve(y_test, probs)
    auprc = auc(r, p)

    beta = 2
    f2_scores = (1 + beta**2) * (p[:-1] * r[:-1]) / ((beta**2 * p[:-1]) + r[:-1] + 1e-8)
    best_idx = np.argmax(f2_scores)
    best_threshold = t[best_idx]

    y_pred = (probs >= best_threshold).astype(int)

    print(f"{model_name} AUPRC: {auprc:.4f}  F2: {f2_scores[best_idx]:.4f}  threshold: {best_threshold:.4f}")

    return {
        "auprc": auprc,
        "best_threshold": best_threshold,
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f2": fbeta_score(y_test, y_pred, beta=2),
    }
