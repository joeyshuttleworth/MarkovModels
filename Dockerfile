FROM python:3.9-buster

COPY requirements.txt /opt/app/requirements.txt

COPY . MarkovModels

RUN apt-get update && apt-get install git graphviz graphviz-dev gcc bash -y
RUN pip install --upgrade pip
RUN pip install -r MarkovModels/requirements.txt

RUN git clone https://github.com/CardiacModelling/markov-builder.git \
    && cd markov-builder \
    && git checkout add_kemp_model \
    && pip install pygraphviz \
    && pip install . \
    && pip install pytest \
    && pytest

RUN mkdir /data

WORKDIR /opt/app/MarkovModels

ENTRYPOINT ["/bin/bash", "-c", "-l"]
CMD ["bash"]
