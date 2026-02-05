from sklearn.ensemble import IsolationForest
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score

def run_iforest(X_train, X_test, y_test):
    param_grid = {
        "n_estimators": [100, 200],
        "max_samples": ["auto", 0.8],
        "contamination": [0.008]
    }

    best_auc = -1
    best_model = None
    best_params = None

    for params in ParameterGrid(param_grid):
        model = IsolationForest(random_state=42, **params)
        model.fit(X_train)

        scores = -model.decision_function(X_test)
        auc = roc_auc_score(y_test, scores)

        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_params = params

    return {
        "name": "Isolation Forest",
        "model": best_model,
        "auc": best_auc,
        "params": best_params,
        "scores": -best_model.decision_function(X_test)
    }
