from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score

def run_ocsvm(X_train, X_test, y_test):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ocsvm", OneClassSVM(kernel="rbf"))
    ])

    param_grid = {
        "ocsvm__nu": [0.001, 0.005, 0.01, 0.05],
        "ocsvm__gamma": ["scale", 0.01, 0.1, 1.0]
    }

    best_auc = -1
    best_model = None
    best_params = None

    for params in ParameterGrid(param_grid):
        model = pipeline.set_params(**params)
        model.fit(X_train)

        scores = -model.decision_function(X_test)
        auc = roc_auc_score(y_test, scores)

        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_params = params

    return {
        "name": "One-Class SVM",
        "model": best_model,
        "auc": best_auc,
        "params": best_params,
        "scores": -best_model.decision_function(X_test)
    }
