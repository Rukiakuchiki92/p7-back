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


# lancement server uvicorn
python uvicorn main_copy:app --reload

# à quoi sert postman

sert tester plus facilement une api en lui envoyant des requêtes HTTP 