FROM python:3.9-slim

WORKDIR /opt/app
COPY requirements.txt /opt/app/requirements.txt

RUN apt-get update && apt-get install git graphviz gcc -y

RUN git clone https://github.com/CardiacModelling/markov-builder.git \
    && cd markov-builder \
    && pip install . \
    && pip install pytest \
    && pytest
