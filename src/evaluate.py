from sklearn.metrics import precision_recall_curve, auc
import numpy as np
from sklearn.metrics import precision_score, recall_score, fbeta_score




def evaluate_model(model, X_test, y_test, model_name: str) -> float:
    """Compute AUPRC"""
    probs = model.predict_proba(X_test)[:,1]

    

    p,r,t = precision_recall_curve(y_test, probs)
    print("precision", p)
    print("recall", r)
    print("threshold", t)

    auprc = auc(r, p)
    print(f"{model_name} AUPRC: {auprc:.4f}")

    beta = 2

    f2_scores = (1 + beta**2) * (p[:-1] * r[:-1]) / (
        (beta**2 * p[:-1]) + r[:-1] + 1e-8
    )

    best_idx = np.argmax(f2_scores)

    print(f"Best F2: {f2_scores[best_idx]:.4f}")
    print(f"Best threshold: {t[best_idx]:.4f}")

    y_pred = (probs >= t[best_idx]).astype(int)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f2 = fbeta_score(y_test, y_pred, beta=2)

    print(f"Final Precision: {precision:.4f}")
    print(f"Final Recall: {recall:.4f}")
    print(f"Final F2: {f2:.4f}")

    return auprc



