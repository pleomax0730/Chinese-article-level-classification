FROM python:3.9-slim-bullseye

ENV LANG C.UTF-8

RUN apt-get update && \
    apt-get install -y bash \
    build-essential \
    curl \
    libsndfile1-dev \
    ca-certificates \
    libpython3.9-dev \
    python3.9-dev \
    python3.9-venv


RUN python -m pip install --no-cache-dir --upgrade pip


WORKDIR /home/ubuntu/HSK3

COPY requirements.txt requirements.txt

RUN pip3 install --no-cache-dir -r requirements.txt

RUN pip3 install --upgrade setuptools uwsgi

EXPOSE 8030

COPY . .

CMD ["bash", "script.sh"]
