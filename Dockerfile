FROM mambaorg/micromamba

USER root

ARG UID=1001

COPY requirements.txt /opt/app/requirements.txt

COPY . /markovmodels
COPY environment.yml /markovmodels/environment.yml

WORKDIR /markovmodels

RUN useradd --uid ${UID} --create-home toto_user

RUN apt-get update && apt-get upgrade -y \
    && apt-get install git graphviz graphviz-dev gcc bash build-essential cmake gfortran -y \
    && apt-get install texlive-latex-extra texlive-fonts-recommended dvipng cm-super -y \
    && chown -R toto_user . 

USER toto_user
WORKDIR /markovmodels

RUN micromamba env create -f /markovmodels/environment.yml \
    && mkdir -p output

ENTRYPOINT ["micromamba", "run", "-n", "markovmodels"]
CMD ["python", "--version"]
