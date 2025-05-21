# main_prova.py
import os
import sys
import numpy as np
import joblib
# Afegim source a sys.path per trobar pipeline_prova.py
sys.path.append(os.path.join(os.path.dirname(__file__), 'source'))
from pipeline_prova import load_windows
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

if __name__ == '__main__':
    base = 'data_prova'
    dirs = {
        'AFIB': os.path.join(base, 'data_afib'),
        'Healthy': os.path.join(base, 'data_healthy')
    }
    window_sec = 30

    def list_recs(path):
        return sorted({os.path.splitext(f)[0] for f in os.listdir(path) if f.endswith('.dat')})

    X_parts, y_parts = [], []
    for label, d in dirs.items():
        recs = list_recs(d)
        print(f"{label}: {len(recs)} registres a {d}")
        X, y = load_windows(recs, d, window_sec)
        if X.size == 0:
            raise RuntimeError(f"No finestres vàlides de {label} a {d}")
        # Assignem etiquetes clares per carpeta
        if label == 'AFIB':
            y = np.ones_like(y)
        else:  # Healthy
            y = np.zeros_like(y)
        X_parts.append(X)
        y_parts.append(y)

    # Combina dades
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    print(f"Total finestres: {len(y)}, AF={np.sum(y)}, Normals={len(y)-np.sum(y)}")

    # Disseny del pipeline i validació
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
    print(f'AUC 5-fold estratificat: {np.nanmean(scores):.3f}')

    # Entrenament final
    clf.fit(X, y)

    # Desa el model
    os.makedirs('models', exist_ok=True)
    joblib.dump(clf, os.path.join('models', 'rf_af_model.joblib'))
    print('✅ Model entrenat i desat com models/rf_af_model.joblib')
