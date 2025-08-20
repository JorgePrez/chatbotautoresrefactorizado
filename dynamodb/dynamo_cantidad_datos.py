import yaml

with open("userschh_login_google_conteo.yaml", "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

usuarios = data["credentials"]["usernames"]
print("Cantidad de usuarios:", len(usuarios))