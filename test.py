import pytest
import pandas as pd
from fastapi import HTTPException
from unittest.mock import MagicMock
import numpy as np
# On suppose que tu as importé les fonctions depuis ton fichier principal
from main_copy import check_client_exists, get_client_features, get_model_prediction, get_decision_and_commentary, get_warnings

@pytest.fixture
def sample_X_test():
    return pd.DataFrame([
        {"ID_CLIENT": 1, "AGE": 65, "REVENU_TOTAL": 25000, "NBR_ENFANTS": 3, "ANNEES_EMPLOI": 1, "EMPLOI_TYPE": "Retraité"},
        {"ID_CLIENT": 2, "AGE": 35, "REVENU_TOTAL": 40000, "NBR_ENFANTS": 1, "ANNEES_EMPLOI": 5, "EMPLOI_TYPE": "Salarié"}
    ])

@pytest.fixture
def input_cols():
    return ["AGE", "REVENU_TOTAL", "NBR_ENFANTS", "ANNEES_EMPLOI", "EMPLOI_TYPE"]

@pytest.fixture
def dummy_model():
    model = MagicMock()
    model.predict_proba = MagicMock(return_value=np.array([[0.7, 0.3]]))
    return model

### --- TESTS FONCTIONS ---

def test_Given_ClientIdNotInData_When_CheckClientExists_Then_Raise404(sample_X_test):
    with pytest.raises(HTTPException) as excinfo:
        check_client_exists(99, sample_X_test)
    assert excinfo.value.detail == "Client non trouvé"

def test_Given_ClientIdInData_When_CheckClientExists_Then_Pass(sample_X_test):
    # Should not raise
    check_client_exists(1, sample_X_test)

def test_Given_ValidClientId_When_GetClientFeatures_Then_ReturnClientDfAndXInput(sample_X_test, input_cols):
    client_df, X_input = get_client_features(1, sample_X_test, input_cols)
    assert client_df.shape[0] == 1
    assert all(col in X_input.columns for col in input_cols)

def test_Given_XInputAndModel_When_GetModelPrediction_Then_ReturnProb(dummy_model, sample_X_test, input_cols):
    _, X_input = get_client_features(1, sample_X_test, input_cols)
    prob = get_model_prediction(X_input, dummy_model)
    assert prob == 0.3

def test_Given_ProbBelowThreshold_When_GetDecisionAndCommentary_Then_Approve():
    #act
    prob = 0.2
    best_thresh = 0.5
    #arrange
    approved, decision, commentary = get_decision_and_commentary(prob, best_thresh)
    #assert
    assert approved is True
    assert "ACCORDÉ" in decision
    assert "Profil validé" in commentary

def test_Given_ProbAboveThreshold_When_GetDecisionAndCommentary_Then_Refuse():
    prob = 0.8
    best_thresh = 0.5
    approved, decision, commentary = get_decision_and_commentary(prob, best_thresh)
    assert approved is False
    assert "REFUSÉ" in decision
    assert "Attention au risque" in commentary

def test_GivenRetraiteSenior_When_GetWarnings_Then_RetourneAvertissement(sample_X_test):
    client_df = sample_X_test[sample_X_test["ID_CLIENT"] == 1]
    warnings = get_warnings(client_df)
    assert any("Retraité senior" in w for w in warnings)

def test_GivenChargeFamiliale_When_GetWarnings_Then_RetourneAvertissement(sample_X_test):
    client_df = sample_X_test[sample_X_test["ID_CLIENT"] == 1]
    warnings = get_warnings(client_df)
    assert any("Charge familiale" in w for w in warnings)

def test_GivenFaibleRevenu_When_GetWarnings_Then_RetourneAvertissement(sample_X_test):
    client_df = sample_X_test[sample_X_test["ID_CLIENT"] == 1]
    warnings = get_warnings(client_df)
    assert any("Revenu faible" in w for w in warnings)

def test_GivenAncienneteProFaible_When_GetWarnings_Then_RetourneAvertissement(sample_X_test):
    client_df = sample_X_test[sample_X_test["ID_CLIENT"] == 1]
    warnings = get_warnings(client_df)
    assert any("Ancienneté pro faible" in w for w in warnings)


    