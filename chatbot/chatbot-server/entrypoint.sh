#!/bin/bash

python manage.py makemigrations monitor

python manage.py migrate monitor

python manage.py makemigrations 

python manage.py migrate

python manage.py createsuperuser --no-input

export WORKERS=${SERVER_WORKERS:-3}

export TIMEOUT=${WORKER_TIMEOUT:-180}

exec gunicorn chatgpt_ui_server.wsgi --workers=$WORKERS --timeout $TIMEOUT --bind 0.0.0.0:8000 --access-logfile -