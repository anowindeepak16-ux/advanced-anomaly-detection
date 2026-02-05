import numpy as np
from sklearn.model_selection import train_test_split

def generate_data(random_state=42):
    rng = np.random.RandomState(random_state)

    # Dataset configuration
    n_samples = 1000
    n_features = 20
    anomaly_fraction = 0.008  # <1%

    n_anomalies = int(n_samples * anomaly_fraction)
    n_normals = n_samples - n_anomalies

    # Normal data
    X_normal = rng.normal(0, 1, size=(n_normals, n_features))

    # Anomalies (subtle)
    X_anomaly = rng.normal(0.5, 1.2, size=(n_anomalies, n_features))

    # Combine
    X = np.vstack([X_normal, X_anomaly])
    y = np.hstack([np.zeros(n_normals), np.ones(n_anomalies)])

    # Shuffle
    perm = rng.permutation(n_samples)
    X, y = X[perm], y[perm]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=random_state
    )

    # Train only on normal data
    return X_train[y_train == 0], X_test, y_test
