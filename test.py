
from joblib import load
from typing import Any, Hashable
import hashlib
import pandas as pd
import traceback
from fastapi import FastAPI,Form, File, UploadFile
import numpy as np
import shap
import pickle as pk

# Créer une instance de l'application FastAPI
app = FastAPI()


########################
# Lecture des fichiers #
########################


def lecture_x_test_original ():
    x_test_original = pd.read_csv("app_train_sample.csv")
    x_test_original = x_test_original.rename(columns=str.lower)
    return x_test_original


def lecture_x_test_original_clean():
    x_test_clean = pd.read_csv("app_train_sample_clean.csv")
    return x_test_clean



shap_dict = load("test_shap.joblib")

featuretmp = load('feat_importances.joblib')
explainertmp = load('explainer.joblib')
shap_valuestmp = load('shap_values.joblib')
clienttmp = load('client.joblib')
grid1 = load('model_lg.joblib')

#################################################
# Lecture du modèle de prédiction et des scores #
#################################################
model_rf = load("model_rf.joblib")


y_pred_rf = model_rf.predict(lecture_x_test_original_clean().drop(labels="ID_CLIENT", axis=1))    # Prédiction de la classe 0 ou 1
y_pred_rf_proba = model_rf.predict_proba(lecture_x_test_original_clean().drop(labels="ID_CLIENT", axis=1)) # Prédiction du % de risque

# Récupération du score du client
y_pred_proba_df = pd.DataFrame(y_pred_rf_proba, columns=['proba_classe_0', 'proba_classe_1'])
y_pred_proba_df = pd.concat([y_pred_proba_df['proba_classe_1'],lecture_x_test_original_clean()['ID_CLIENT']], axis=1)

# Récupération de la décision
y_pred_rf_df = pd.DataFrame(y_pred_rf, columns=['prediction'])
y_pred_rf_df = pd.concat([y_pred_rf_df, lecture_x_test_original_clean()['ID_CLIENT']], axis=1)
y_pred_rf_df['client'] = np.where(y_pred_rf_df.prediction == 1, "Le client n'est pas solvable 💸🚫  ", "Le client est solvable 💰🥳 ")
y_pred_rf_df['decision'] = np.where(y_pred_rf_df.prediction == 1, "CRÉDIT NON ACCORDÉ 🚫", "CRÉDIT ACCORDÉ 🥳")

# app
app = FastAPI()

@app.get("/predic_client/{id_client}")
def predict(id_client: int):
    all_id_client = list(lecture_x_test_original_clean()['ID_CLIENT'])
    
    ID = id_client
    ID = int(ID)
    if ID not in all_id_client:
        number="L'identifiant que vous avez saisi n'est pas valide !"
        prediction="NA"
        solvabilite="NA"
        decision="NA"
    else :
        number="Identifiant client trouvé"
        score = y_pred_proba_df[y_pred_proba_df['ID_CLIENT']==ID]
        prediction = round(score.proba_classe_1.iloc[0]*100, 1)
        solvabilite = y_pred_rf_df.loc[y_pred_rf_df['ID_CLIENT']==ID, "client"].values
        solvabilite = solvabilite[0]
        decision = y_pred_rf_df.loc[y_pred_rf_df['ID_CLIENT']==ID, "decision"].values
        decision = decision[0]
    liste = [{"number": number, "prediction": prediction, "solvabilite": solvabilite, "decision": decision}]    
    return liste


# Définir la route pour récupérer les informations du client en fonction de son ID
@app.get("/client_info/{client_id}")
def get_client_info(client_id: int):
    # Recherche des informations du client en fonction de son ID dans les données nettoyées
    client_info = lecture_x_test_original_clean()[lecture_x_test_original_clean()['ID_CLIENT'] == client_id].to_dict(orient='records')
    
    # Si le client est trouvé, renvoyer ses informations
    if client_info:
        return client_info[0]
    # Sinon, renvoyer un message d'erreur
    else:
        return {"error": "Client not found"}



@app.get('/Shap/{client_id}')
def client_shap_df(client_id: int):
    # Vérifier si l'ID client existe dans le dataframe
    if client_id not in clienttmp.index:
        return {"error": "Client ID not found"}

    # Extraire les données du client
    client_data = clienttmp.iloc[[client_id]]
    
    # Faire des prédictions pour ce client
    y_pred_proba_list = grid1.predict_proba(client_data)
    y_pred_rf_proba = grid1.predict_proba(client_data)

    # Afficher les informations sur le client
    model_pred = round(y_pred_proba_list[0][0], 4)
    client_number = client_data.iloc[0, 0]  # Supposons que la première colonne contient l'ID
    risques = f"Il y a {y_pred_rf_proba[0][1]:.1%} de risques que le client ait des difficultés de paiement"

    # Préparer les valeurs SHAP
    client_number_cleaned = int(client_number) if isinstance(client_number, (np.integer, np.int64)) else client_number
    client_shap_values = shap_valuestmp[client_id].values  # Correction de l'indexation

    # Si les valeurs SHAP sont complexes, les convertir en liste ou dict
    if isinstance(client_shap_values, np.ndarray):
        client_shap_values = client_shap_values.tolist()  # Convertir en liste
    elif isinstance(client_shap_values, shap.Explanation):
        client_shap_values = client_shap_values.data  # Extraire les données pertinentes
    # Préparer le dictionnaire des informations du client
    client_info = {
        "client_id": client_number_cleaned,
        "model_prediction": model_pred,
        "risques": risques,
        "shap_client": client_shap_values,
        "featuresimp": shap_valuestmp.feature_names
    }

    return client_info


@app.get('/recup_index/{client_id}')
def get_index(client_id: int):
    all_client_ids = lecture_x_test_original_clean()['ID_CLIENT'].tolist()

    if client_id not in all_client_ids:
        return {"error": "Client's ID not found"}

    client_data_index = lecture_x_test_original_clean()[lecture_x_test_original_clean()['ID_CLIENT'] == client_id].index[0]

    return client_data_index










#VRAI MAIN.PY


from fastapi import FastAPI
from joblib import load
import pandas as pd
import numpy as np
import shap

# Init FastAPI app
app = FastAPI()

# Chargement des objets sauvegardés
model = load("model_logistic_regression.joblib")

preprocessor = load("preprocessor.joblib")
explainer = load("explainer_logreg.joblib")
best_thresh = load("best_thresh_logreg.joblib")
X_test_eval = load("X_test_eval.joblib")
feature_names = load("feature_names.joblib")
clienttmp = load('X_test_eval.joblib')

# ========================================
# ================ ROUTES ================
# ========================================


# Infos client
@app.get("/client_info/{client_id}")
def get_client_info(client_id: int):
    client_info = X_test_eval[X_test_eval["ID_CLIENT"] == client_id]
    if client_info.empty:
        return {"error": "Client non trouvé"}
    return client_info.iloc[0].to_dict()

# Index du client
@app.get("/recup_index/{client_id}")
def get_index(client_id: int):
    if client_id not in X_test_eval["ID_CLIENT"].values:
        return {"error": "ID client introuvable"}
    idx = X_test_eval[X_test_eval["ID_CLIENT"] == client_id].index[0]
    return {"index": int(idx)}

# Prédiction binaire avec seuil métier
@app.get("/predic_client/{id_client}")
def predict(id_client: int):
    client = X_test_eval[X_test_eval["ID_CLIENT"] == id_client]

    if client.empty:
        return {"error": "ID client non trouvé."}

    idx = client.index[0]
    features = client[feature_names]
    prob = model.predict_proba(features)[0][1]
    prediction = int(prob >= best_thresh)
    decision = "CRÉDIT REFUSÉ 🚫" if prediction == 1 else "CRÉDIT ACCORDÉ 🥳"
    commentaire = "Client non solvable 💸🚫" if prediction == 1 else "Client solvable 💰🥳"

    return {
        "id_client": id_client,
        "probabilité défaut (%)": round(prob * 100, 1),
        "décision": decision,
        "commentaire": commentaire
    }

# Valeurs SHAP du client
@app.get("/Shap/{client_id}")
def get_shap_values(client_id: int):
    client = X_test_eval[X_test_eval["ID_CLIENT"] == client_id]
    if client.empty:
        return {"error": "Client ID non trouvé"}

    features = client[feature_names]
    transformed = model.named_steps["preprocessor"].transform(features)
    shap_values = explainer(transformed)

    return {
        "client_id": int(client_id),
        "shap_values": shap_values[0].values.tolist(),
        "feature_names": feature_names.tolist()
    }


# 🧠 Top 3 raisons SHAP du refus ou accord
@app.get("/explication_rapide/{client_id}")
def explication_rapide(client_id: int):
    client = X_test_eval[X_test_eval["ID_CLIENT"] == client_id]
    if client.empty:
        return {"error": "Client non trouvé"}

    features = client[feature_names]
    transformed = model.named_steps["preprocessor"].transform(features)
    shap_values = explainer(transformed)[0]

    top_features = sorted(
        zip(shap_values.values, feature_names),
        key=lambda x: abs(x[0]),
        reverse=True
    )[:3]

    return {
        "client_id": client_id,
        "principales_raisons": [
            {"feature": feat, "impact": round(val, 4)} for val, feat in top_features
        ]
    }
