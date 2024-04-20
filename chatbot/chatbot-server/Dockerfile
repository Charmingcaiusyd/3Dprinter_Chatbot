FROM python:3.10-slim as wsgi-server

RUN apt update && apt install -y --no-install-recommends \
    python3-dev default-libmysqlclient-dev build-essential libpq-dev \
    dos2unix libgl1-mesa-glx libglib2.0-0 pkg-config libcairo2-dev \
    libsystemd-dev \
    && rm -rf /var/lib/apt/lists/*

ENV DJANGO_SUPERUSER_USERNAME=admin
ENV DJANGO_SUPERUSER_PASSWORD=password
ENV DJANGO_SUPERUSER_EMAIL=admin@example.com

COPY requirements.txt ./

RUN pip install --no-cache-dir --default-timeout=100 -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
RUN pip install django-apscheduler

WORKDIR /app

COPY . .


RUN /bin/sh -c python manage.py makemigrations monitor
RUN /bin/sh -c python manage.py migrate monitor
RUN /bin/sh -c python manage.py makemigrations 
RUN /bin/sh -c python manage.py migrate
RUN /bin/sh -c python manage.py migrate django_apscheduler



RUN /bin/sh -c python manage.py check --deploy \
    && /bin/sh -c python manage.py collectstatic --no-input \
    && dos2unix entrypoint.sh \
    && chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]

EXPOSE 8000


FROM nginx:1.22-alpine as web-server

COPY nginx.conf /etc/nginx/templates/default.conf.template
