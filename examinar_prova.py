# infer.py
import joblib
import numpy as np

FEATURES = ['meanRR', 'stdRR', 'RMSSD', 'pNN50']

# def carregar_model(path='models/rf_af_model.joblib'):
def carregar_model(path='models/rf_af_model2.joblib'):
    return joblib.load(path)

def prediu_af(model, vector):
    """
    vector: llista o np.array de 4 valors [meanRR, stdRR, RMSSD, pNN50]
    Retorna (etiqueta, probabilitat)
    """
    x = np.array(vector).reshape(1, -1)
    et = model.predict(x)[0]
    pr = model.predict_proba(x)[0,1]
    return et, pr

if __name__ == '__main__':
    model = carregar_model()
    # Ex.: aquests valors els extindries abans amb pipeline de feature-extraction
    mostra = [0.8, 0.1, 0.08, 25.0]
    et, pr = prediu_af(model, mostra)
    label_str = 'AF' if et==1 else 'Normal'
    print(f'Resultat: {label_str} (P(Afib) = {pr:.2f})')
