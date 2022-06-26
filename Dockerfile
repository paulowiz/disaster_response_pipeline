FROM python:3.8-slim-buster

LABEL maintainer="Paulo Mota <phmota@outlook.com.br>"

RUN apt-get update
RUN apt-get install -y python3-pip

COPY . .

RUN pip install --no-cache-dir -r requirements.txt
RUN python3 data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_project.db
RUN python3 models/train_classifier.py data/disaster_project.db models/classifier.pkl
EXPOSE 80

CMD ["uvicorn"  , "app.main:app", "--host", "0.0.0.0", "--port", "80","--workers","2","--reload"]