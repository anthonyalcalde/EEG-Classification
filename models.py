"""
Classifier definitions and training helpers.
"""
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import config

MODELS = {
    "LDA": LinearDiscriminantAnalysis(),
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=config.RANDOM_STATE),
    "SVM (RBF)": SVC(kernel="rbf", C=1.0, probability=True, random_state=config.RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=config.RANDOM_STATE),
    "k-NN": KNeighborsClassifier(n_neighbors=5),
}


def train_all(X_train, y_train):
    """Fit a fresh copy of every model in MODELS on the training set."""
    fitted = {}
    for name, model in MODELS.items():
        m = clone(model)
        m.fit(X_train, y_train)
        fitted[name] = m
    return fitted
