# main_copy.py (API)
from fastapi import FastAPI, HTTPException
from joblib import load
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)
app = FastAPI(title="API Crédit - Business Rules & SHAP")

# Chargement des artefacts
model = load("model_logistic_regression.joblib")  # pipeline complet incluant preprocess
best_thresh = float(load("best_thresh_logreg.joblib"))
explainer = load("explainer_logreg.joblib")
X_test = load("X_test_eval.joblib")  # DataFrame avec colonne ID_CLIENT et features
feature_names = load("feature_names.joblib")

# Colonnes d'entrée : toutes sauf ID_CLIENT et CIBLE (si présente)
input_cols = [c for c in X_test.columns if c not in ['ID_CLIENT', 'CIBLE']]

def check_client_exists(client_id, X_test):
    if client_id not in X_test["ID_CLIENT"].values:
        raise HTTPException(status_code=404, detail="Client non trouvé")

def get_client_features(client_id, X_test, input_cols):
    client_df = X_test[X_test["ID_CLIENT"] == client_id]
    X_input = client_df[input_cols]
    return client_df, X_input

def get_model_prediction(X_input, model):
    prob = model.predict_proba(X_input)[0, 1]
    return prob

def get_decision_and_commentary(prob, best_thresh):
    approved = prob < best_thresh
    decision = "CRÉDIT ACCORDÉ 🥳" if approved else "CRÉDIT REFUSÉ 🚫"
    commentary = (
        "Client solvable 💰 Profil validé" if approved else
        "Client non solvable 💸 Attention au risque"
    )
    return approved, decision, commentary

def get_warnings(client_df):
    warnings = []
    age = float(client_df["AGE"].iloc[0])
    revenu = float(client_df["REVENU_TOTAL"].iloc[0])
    enfants = int(client_df["NBR_ENFANTS"].iloc[0])
    anciennete = float(client_df['ANNEES_EMPLOI'].iloc[0])
    emploi = client_df["EMPLOI_TYPE"].iloc[0]
    if emploi in ("Retraité", "Pensioner") and age > 63:
        warnings.append("🕰️ Retraité senior : vérifier fin droit pension & assurances")
    if enfants >= 3:
        warnings.append("👨‍👩‍👧‍👦 Charge familiale élevée : adapter montant crédit")
    if revenu < 30000:
        warnings.append("💸 Revenu faible : exiger caution ou co-emprunteur")
    if anciennete < 2:
        warnings.append("📈 Ancienneté pro faible : proposer durée plus courte")
    return warnings


@app.get("/clients")
def get_all_clients():
    return {"clients": X_test["ID_CLIENT"].tolist()}

@app.get("/get_data/")
def get_data():
    df = X_test.reset_index(drop=True)
    return df.to_dict(orient="records")

@app.get("/predict/{client_id}")
def predict_client(client_id: int):
    try:
        check_client_exists(client_id, X_test)
        client_df, X_input = get_client_features(client_id, X_test, input_cols)
        prob = get_model_prediction(X_input, model)
        approved, decision, commentary = get_decision_and_commentary(prob, best_thresh)
        warnings = get_warnings(client_df)
        return {
            "id_client": client_id,
            "probabilité_défaut": round(prob * 100, 1),
            "décision": decision,
            "commentaire": commentary,
            "warnings": warnings
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur interne predict_client {client_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {e}")

@app.get("/shap/{client_id}")
def get_shap_values(client_id: int):
    if client_id not in X_test["ID_CLIENT"].values:
        raise HTTPException(status_code=404, detail="Client non trouvé")
    client_df = X_test[X_test["ID_CLIENT"] == client_id]
    # Préprocessing cohérent avec le modèle
    X_input = client_df[input_cols]
    Xt = model.named_steps['preprocessor'].transform(X_input)
    shap_vals = explainer(Xt)[0]
    return {
        "client_id": client_id,
        "shap_values": shap_vals.values.tolist(),
        "feature_names": feature_names.tolist()
    }

@app.get("/top_reasons/{client_id}")
def get_top_reasons(client_id: int):
    out = get_shap_values(client_id)
    vals = np.array(out["shap_values"])
    idx = np.argsort(-np.abs(vals))[:3]
    reasons = [{"feature": feature_names[i], "impact": float(vals[i])} for i in idx]
    return {"client_id": client_id, "top_reasons": reasons}



