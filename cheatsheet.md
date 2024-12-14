# command utile pour amel

``` bash
python -m venv venv # Create virtual environment
source venv/Scripts/activate # Activate virtual environment
pip install pandas # Install your packages
pip freeze > requirements.txt # Freeze packages to text file
deactivate # Deactivate venv
```
 
``` powershell
pip list --outdated | Select-Object -Skip 2 | Select-String "^\S+" | ForEach-Object { pip install --upgrade $_.Matches[0].Value } # mettre à jours tout tes packages python en powershell
```

# command git utile

``` bash
git init # Initialize repo
echo 'venv' > .gitignore # Add venv to .gitignore
git add .
git commit -m 'initial commit'
git push
```

# url address site 

http://127.0.0.1:8000

https://stackoverflow.com/questions/59025891/uvicorn-is-not-working-when-called-from-the-terminal

# lancement server uvicorn
python -m uvicorn main_copy:app --reload

# à quoi sert postman

sert tester plus facilement une api en lui envoyant des requêtes HTTP 




@app.get('/Shap/{client_id}')
def client_shap_df(client_id: int):
    all_client_ids = lecture_x_test_original_clean()['ID_CLIENT'].tolist()

    if client_id not in all_client_ids:
        return {"error": "Client's ID not found"}

    client_data_index = lecture_x_test_original_clean()[lecture_x_test_original_clean()['ID_CLIENT'] == client_id].index[0]

    shap_values = shap_dict[client_data_index]

    shap_json = {"shap_values": shap_values}

    return shap_json





@app.get('/feature_importances/')
def get_feature_importances():
    feat_imp_df = pd.DataFrame(feat_imp, columns=['feature', 'importance'])
    feat_imp_json = feat_imp_df.to_dict(orient='records')
    return feat_imp_json