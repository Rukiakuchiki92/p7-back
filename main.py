"""
Main file of the API
"""

import json
import pickle as pk
from typing import Any, Hashable
import hashlib
import pandas as pd
import traceback
from fastapi import FastAPI, File, UploadFile

# data
df = pd.read_csv("app_train_sample_clean.csv")


###################################
# TODO : load shap values object
###################################


###################################
# TODO : load model ML
###################################


def manual_fillna(value):
    """ """

    if isinstance(value, (int, float, str)):

        if isinstance(value, float):
            return round(value, 4)
        return value

    return f"NAN : {str(type(value))}"


def convert_dictionary(original_dict: dict[Hashable, Any]) -> dict[str, Any]:
    """fonction permettant de convertir un dictionnaire de type dict[hashable,any] -> dict[str, Any]

    :param  original_dict: dictionnaire d'entrée de la fonction

    returns dict[str, Any]
    """

    new_dict = {str(key): value for key, value in original_dict.items()}

    return new_dict


def convert_dict_to_list(input_dict: dict[Hashable, Any]) -> list[tuple[str, Any]]:
    """fonction permettant de convertir un dictionnaire de type dict[hashable,any] -> list[tuple[str, Any]]

    :param  original_dict: dictionnaire d'entrée de la fonction

    returns list[tuple[str, Any]]
    """

    result_list = [(str(key), value) for key, value in input_dict.items()]

    return result_list


# app
app = FastAPI()


@app.get("/")
def root():
    """Home page"""

    return {"Hello": "World"}


@app.get("/get_list_ids")
def get_list_ids():
    """Return list of ids"""

    return {"list_ids": df["ID_CLIENT"].tolist()}


@app.get("/get_population_summary/")
def get_population_summary():
    """Return population summary"""

    # select_columns = ["AGE", "REVENU_TOTAL", "CNT_FAM_MEMBERS"]

    select_data = df.iloc[:, :20].describe().round(2).to_dict()
    list_population = convert_dictionary(select_data)

    return {"population_summary": list_population}


@app.get("/get_client_info/{client_id}")
def get_client_info(client_id: int):
    """Return data dict for a client"""

    select_row = df.loc[df["ID_CLIENT"] == client_id].to_dict()
    client_info = convert_dictionary(select_row)

    client_info = {k: manual_fillna(v) for k, v in client_info.items()}

    return {"client_info": client_info}


@app.get("/get_prediction/{client_id}")
def get_prediction(client_id):
    """Return prediction for a client"""

    # load the df
    # find the client with his id
    # transform if needed the vector client
    # perform the .predict of this client
    # return the prediction

    ###################
    # TODO : code this
    ###################

    client_predit = {
        "0": 0.55,
        "1": 0.45,
    }

    return {"client_predit": client_predit}

def compute_sha1(file_content):
    sha1 = hashlib.sha1()
    sha1.update(file_content)
    return sha1.hexdigest()

@app.get("/get_shap/")
async def get_shap():
    """Return shap values for a client"""
    try:
            
            # Load the pickled data
        with open("feat_importance.pk", "rb") as f:
            contents = f.read()


        loaded_data = pk.loads(contents)
        return {"status": "success", "file_content": loaded_data}
    except FileNotFoundError:
        return {"status": "error", "message": "Fichier introuvable."}
    except pk.UnpicklingError as e:
        return {"status": "error", "message": "Erreur lors du chargement des données picklées : " + str(e)}
    except Exception as e:
        return {"status": "error", "message": traceback.format_exc()}
