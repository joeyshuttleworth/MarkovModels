FROM continuumio/miniconda3

ARG UID=1001

COPY requirements.txt /opt/app/requirements.txt

COPY . MarkovModels
WORKDIR /MarkovModels

RUN useradd --uid ${UID} toto_user

RUN apt-get update && apt-get install git graphviz graphviz-dev gcc bash build-essential cmake gfortran -y
RUN apt-get install texlive-latex-extra texlive-fonts-recommended dvipng cm-super -y

RUN conda env create -f environment.yml
SHELL [ "conda", "run", "-n", "markovmodels", "/bin/bash", "-c"]

RUN pip install .

RUN mkdir output
RUN chown -R toto_user .
USER toto_user

ENTRYPOINT ["/bin/bash", "-c", "-l"]
CMD ["bash"]
    