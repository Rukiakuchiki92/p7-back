from fastapi import FastAPI
from joblib import load
import pandas as pd
import numpy as np
import shap

# Initialisation de l'application
app = FastAPI()

#####################################
# Chargement des fichiers nécessaires
#####################################

model = load("model_logistic_regression.joblib")
preprocessor = load("preprocessor.joblib")
explainer = load("explainer_logreg.joblib")
best_thresh = load("best_thresh_logreg.joblib")
X_test_eval = load("X_test_eval.joblib")
feature_names = load("feature_names.joblib")

#####################################
# ROUTES PRINCIPALES DE L'API
#####################################
@app.get("/clients")
def list_clients():
    return {"clients": X_test_eval["ID_CLIENT"].tolist()}

# 1. Infos client
@app.get("/client_info/{client_id}")
def get_client_info(client_id: int):
    client_info = X_test_eval[X_test_eval["ID_CLIENT"] == client_id]
    if client_info.empty:
        return {"error": "Client non trouvé"}
    return client_info.iloc[0].to_dict()


# 2. Prédiction avec seuil métier
@app.get("/predict/{client_id}")
def predict_client(client_id: int):
    client = X_test_eval[X_test_eval["ID_CLIENT"] == client_id]
    if client.empty:
        return {"error": "ID client non trouvé"}

    # 1. Transformation et prédiction
    colonnes_origine = model.named_steps["preprocessor"].feature_names_in_
    client_features = client[colonnes_origine]
    transformed = model.named_steps["preprocessor"].transform(client_features)
    prob = model.named_steps["model"].predict_proba(transformed)[0][1]
    threshold = best_thresh
    decision_flag = prob >= threshold

    # 2. Texte de décision
    if decision_flag:
        decision_text = "CRÉDIT REFUSÉ 🚫"
        commentaire = "Client non solvable 💸🚫"
    else:
        decision_text = "CRÉDIT ACCORDÉ 🥳"
        commentaire = "Client solvable 💰🥳"

    # 3. Règles métier → warnings
    warnings = []
    age     = int(client["AGE"].iloc[0])
    revenus = float(client["REVENU_TOTAL"].iloc[0])
    enfants = int(client["NBR_ENFANTS"].iloc[0])
    statut  = client["EMPLOI_TYPE"].iloc[0]  # ou TYPE_REVENUS, TYPE_CONTRAT…
    anciennete_pro = int(client['ANNEES_EMPLOI'].iloc[0])

    if statut == "Retraité" and age > 63 :
        warnings.append("✅ Retraité : vérifier la date de fin de pension et la durée du prêt.")

    if enfants >= 3:
        warnings.append("⚠️ Famille nombreuse : la charge familiale est plus élevée.")
        
    if revenus < 30000:
        warnings.append("⚠️ Revenu faible : confirmer la solvabilité sur les 12 prochains mois.")

    if anciennete_pro >= 5:
        warnings.append("⚠️ Profil professionnel Sénior : Vérifier les points suivants : "
        "- 🩸​🩺​🩻​​ Bilan médical"
        "- ​🗃️​🪪​🗂️​ Assurance vie"
        "- 👴​👵​💰​ Revenu de la pension de retraite."
        "- 🏦​💶​📄​ Historique des crédits si existants")

    if statut not in ("Retraité", 'Pensioner') and anciennete_pro <=3 :
        warnings.append("⚠️ Profil professionnel Junior : Refuser le crédit ou proposer :"
        " Montant du crédit inférieur au montant souhaité par le client "
        "Taux d'intérêt plus élevé (~ 5%)")

    return {
        "id_client": client_id,
        "probabilité_défaut (%)": round(prob * 100, 1),
        "décision": decision_text,
        "commentaire": commentaire,
        "warnings": warnings
    }


# 3. Valeurs SHAP du client
@app.get("/shap/{client_id}")
def get_shap_values(client_id: int):
    client = X_test_eval[X_test_eval["ID_CLIENT"] == client_id]
    if client.empty:
        return {"error": "Client ID non trouvé"}

    # Étape 1 : colonnes originales pour le préprocessing
    colonnes_origine = model.named_steps["preprocessor"].feature_names_in_
    client_features = client[colonnes_origine]

    # Étape 2 : transformation
    transformed = model.named_steps["preprocessor"].transform(client_features)

    # Étape 3 : valeurs SHAP
    shap_values = explainer(transformed)[0]
    print("DEBUG shap len:", len(shap_values.values))
    print("DEBUG feat len:", len(feature_names))
    return {
        "client_id": int(client_id),
        "shap_values": shap_values.values.tolist(),
        "feature_names": feature_names.tolist()
    }


# 4. Top 3 raisons SHAP (positives ou négatives)
@app.get("/explication_rapide/{client_id}")
def get_top_shap_reasons(client_id: int):
    client = X_test_eval[X_test_eval["ID_CLIENT"] == client_id]
    if client.empty:
        return {"error": "Client non trouvé"}

    # Étape 1 : sélection des colonnes d’origine attendues par le preprocessor
    colonnes_origine = model.named_steps["preprocessor"].feature_names_in_
    client_features = client[colonnes_origine]

    # Étape 2 : transformation
    transformed = model.named_steps["preprocessor"].transform(client_features)
    shap_values = explainer(transformed)[0]
    



# 5. Index du client dans X_test
@app.get("/recup_index/{client_id}")
def get_index(client_id: int):
    if client_id not in X_test_eval["ID_CLIENT"].values:
        return {"error": "Client ID introuvable"}
    idx = X_test_eval[X_test_eval["ID_CLIENT"] == client_id].index[0]
    return {"index": int(idx)}
