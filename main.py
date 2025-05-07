import sys
import os

# Afegim 'source/' al path per poder importar pipeline.py
sys.path.append(os.path.join(os.path.dirname(__file__), 'source'))

print("Paths actius:", sys.path)
print("Fitxers disponibles a 'source':", os.listdir(os.path.join(os.path.dirname(__file__), 'source')))

import numpy as np  # ✅ Importem numpy
try:
    from pipeline import load_windows, evaluate_model, train_model
except ModuleNotFoundError as e:
    print(f"❌ Error: {e}. Ensure 'source/pipeline.py' exists and is correctly placed.")
    sys.exit(1)
import joblib

def main():
    records = ["04043", "04015", "04048", "04126", "04746"]
    db_path = "data/MIT-BIH_afdb"  # ✅ Definim db_path dins de main()

    X_all = []
    y_all = []

    for record in records:
        try:
            X, y = load_windows(record, db_path)
            X_all.extend(X)
            y_all.extend(y)
            print(f"✅ {record}: {len(X)} finestres carregades.")
        except Exception as e:
            print(f"❌ Error carregant {record}: {e}")

    X_all = np.array(X_all)
    y_all = np.array(y_all)

    print(f"📊 Total: {len(X_all)} finestres carregades.")

    print("📊 Avaluant model amb validació creuada...")
    clf = train_model(X_all, y_all)
    evaluate_model(X_all, y_all)

    model_path = "models/rf_af_model.joblib"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"💾 Model guardat a: {model_path}")

if __name__ == "__main__":
    main()
