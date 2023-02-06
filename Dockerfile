FROM python:3.9-buster

ARG UID=1001

COPY requirements.txt /opt/app/requirements.txt

COPY . MarkovModels

RUN apt-get update && apt-get install git graphviz graphviz-dev gcc bash -y

RUN useradd --uid ${UID} --create-home toto_user

RUN pip install --upgrade pip

WORKDIR /MarkovModels
RUN pip install -e .

RUN mkdir output
RUN chown -R toto_user /MarkovModels
USER toto_user

ENTRYPOINT ["/bin/bash", "-c", "-l"]
CMD ["bash"]
