# <h1 align="center"> ML_web_HSK3 </h1>

## Environment

- Python 3.9

```bash=
python3.9 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python app.py
```

## Docker

```bash=
sudo docker build --no-cache -t "ponddy/hsk3_level_predictor" -f Dockerfile .
sudo docker run --name ponddy_hsk3_level_predictor_web -d -p 8280:8030 ponddy/hsk3_level_predictor:latest
sudo docker tag image_id ponddy/hsk3_level_predictor:v1.0
sudo docker push ponddy/hsk3_level_predictor:v1.0
sudo docker push ponddy/hsk3_level_predictor:latest
```
