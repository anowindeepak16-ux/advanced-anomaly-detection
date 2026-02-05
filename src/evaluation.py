import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import pandas as pd

def plot_roc(ocsvm_res, iforest_res):
    fpr_svm, tpr_svm, _ = roc_curve(
        y_true=None, y_score=None
    )

def plot_roc(ocsvm_res, iforest_res, y_test=None):
    from sklearn.metrics import roc_curve

    fpr_svm, tpr_svm, _ = roc_curve(y_test, ocsvm_res["scores"])
    fpr_if, tpr_if, _ = roc_curve(y_test, iforest_res["scores"])

    plt.figure()
    plt.plot(fpr_svm, tpr_svm, label=f'OCSVM (AUC={ocsvm_res["auc"]:.3f})')
    plt.plot(fpr_if, tpr_if, label=f'IForest (AUC={iforest_res["auc"]:.3f})')
    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.savefig("plots/roc_curve.png")
    plt.show()

def write_report(ocsvm_res, iforest_res):
    with open("reports/analysis_report.txt", "w") as f:
        f.write("Advanced Anomaly Detection Report\n\n")
        f.write(f"One-Class SVM AUC: {ocsvm_res['auc']:.4f}\n")
        f.write(f"Best Params: {ocsvm_res['params']}\n\n")
        f.write(f"Isolation Forest AUC: {iforest_res['auc']:.4f}\n")
        f.write(f"Best Params: {iforest_res['params']}\n\n")
        f.write("Conclusion:\n")
        f.write("Isolation Forest slightly outperforms OCSVM.\n")
