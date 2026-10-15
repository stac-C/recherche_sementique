# TP — Moteur de recherche sémantique (Django + Ollama)

## Démarrage local

    ollama pull bge-m3
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    python manage.py migrate
    python manage.py runserver      # http://127.0.0.1:8000

Par défaut, Django se connecte à Ollama sur `http://localhost:11434`.

## Connexion à un serveur Ollama distant

Sur la machine qui héberge Ollama, téléchargez le modèle et faites écouter Ollama sur l’interface réseau utilisée par Django :

    ollama pull bge-m3
    $env:OLLAMA_HOST = "0.0.0.0:11434"
    ollama serve

`OLLAMA_HOST` est la configuration du serveur Ollama. Autorisez le port `11434` uniquement depuis la machine Django ou le réseau privé concerné.

Sur la machine Django, définissez l’URL du serveur avant de lancer l’application :

    $env:OLLAMA_URL = "http://192.168.1.50:11434"
    python manage.py runserver

Remplacez `192.168.1.50` par l’adresse IP ou le nom DNS joignable du serveur. Pour l’hébergement Ollama avec authentification par jeton Bearer (ou un proxy distant qui l’exige), définissez aussi :

    $env:OLLAMA_API_KEY = "votre-jeton"

Paramètres optionnels : `OLLAMA_TIMEOUT` (délai des embeddings en secondes, 300 par défaut), `OLLAMA_HEALTH_TIMEOUT` (délai du contrôle de connexion, 5 secondes par défaut) et `EMBED_MODEL` (`bge-m3` par défaut). Ces variables doivent être définies dans l’environnement du processus Django. Ne publiez pas un serveur Ollama non authentifié sur Internet ; préférez un VPN/réseau privé ou un proxy HTTPS authentifié.

L’application utilise d’abord l’API d’embedding actuelle d’Ollama (`/api/embed`). Si le serveur renvoie une erreur 404 sur cette route, elle réessaie avec l’ancienne API (`/api/embeddings`) pour rester compatible avec les versions plus anciennes d’Ollama.

Tests sans Ollama (embeddings factices) : `python manage.py test`
