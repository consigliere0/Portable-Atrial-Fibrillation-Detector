import sys, os
import numpy as np
import joblib

# Afegim el path al pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), 'source'))
from pipeline import load_windows
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

def evaluate_and_train(X, y):
    unique, counts = np.unique(y, return_counts=True)
    print("Distribució de classes:", dict(zip(unique, counts)))
    if len(unique) < 2:
        print("⚠️ No hi ha prou varietat de classes per entrenar/avaluar.")
        return None

    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
    print(f'AUC mitjana (5-fold estrat.): {np.nanmean(scores):.3f}')

    clf.fit(X, y)
    return clf

def main():
    records = ["04126", "04746"]
    db_path = "data_prova"
    window_sec = 10
    min_af_sec = 0.0

    X_all, y_all = [], []
    for rec in records:
        X, y = load_windows(rec, db_path, window_sec, min_af_sec)
        print(f"{rec}: {len(X)} finestres, AF positives: {sum(y)}")
        X_all.extend(X)
        y_all.extend(y)

    X_all = np.array(X_all); y_all = np.array(y_all)
    print(f"TOTAL finestres: {len(X_all)}")

    clf = evaluate_and_train(X_all, y_all)
    if clf:
        os.makedirs("models", exist_ok=True)
        joblib.dump(clf, "models/rf_af_model.joblib")
        print("✅ Model guardat a models/rf_af_model.joblib")

if __name__ == "__main__":
    main()
