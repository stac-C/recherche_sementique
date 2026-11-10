#!/bin/bash
set -e

# Install build dependencies in an isolated environment to avoid modifying
# the system Python managed by the Vercel build image.
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python manage.py collectstatic --noinput --clear
#d