FROM mambaorg/micromamba

ARG UID=1001

COPY requirements.txt /opt/app/requirements.txt

COPY . markovmodels
WORKDIR /markovmodels

RUN useradd --uid ${UID} --create-home toto_user

RUN apt-get update && apt-get upgrade -y \
    && apt-get install git graphviz graphviz-dev gcc bash build-essential cmake gfortran -y \
    && micromamba upgrade -n base -c defaults micromamba \
    && apt-get install texlive-latex-extra texlive-fonts-recommended dvipng cm-super -y \
    && chown -R toto_user . \
    && micromamba env create --prefix /envs/markovmodels -f environment.yml \
    && mkdir /markovmodels/output \
    && micromamba run -p /envs/markovmodels pip install . \
    && chown -R toto_user /envs

USER toto_user

ENTRYPOINT ["micromamba", "run", "--no-capture-output", "-p", "/envs/markovmodels"]
CMD ["python", "--version"]
