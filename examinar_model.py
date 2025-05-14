import joblib
import numpy as np

def carregar_model(path='models/rf_af_model.joblib'):
    print("🔄 Carregant model...")
    model = joblib.load(path)
    print("✅ Model carregat correctament.\n")
    return model

def mostrar_info_model(model):
    print("🔍 Pipeline complet:")
    print(model)

    rf = model.named_steps['rf']
    print("\n🎛️ Paràmetres del Random Forest:")
    for k, v in rf.get_params().items():
        print(f"{k}: {v}")

    print("\n📊 Importància de característiques:")
    features = ['meanRR', 'stdRR', 'RMSSD', 'pNN50']
    for feat, imp in zip(features, rf.feature_importances_):
        print(f"{feat}: {imp:.4f}")

def prediccio_exemple(model):
    print("\n🔮 Predicció d'exemple:")
    # Vector artificial (substitueix per un real si en tens un)
    mostra = np.array([[0.8, 0.1, 0.08, 25.0]])
    pred = model.predict(mostra)[0]
    proba = model.predict_proba(mostra)[0][1]

    print(f"Resultat: {'AF' if pred == 1 else 'Normal'} (probabilitat AF: {proba:.2f})")

def main():
    model = carregar_model()
    mostrar_info_model(model)
    prediccio_exemple(model)

if __name__ == "__main__":
    main()
