FROM continuumio/miniconda3

ARG UID=1001

COPY requirements.txt /opt/app/requirements.txt

COPY . MarkovModels
WORKDIR /MarkovModels

RUN useradd --uid ${UID} --create-home toto_user

RUN apt-get update && apt-get install git graphviz graphviz-dev gcc bash build-essential cmake gfortran -y \
    && apt-get install texlive-latex-extra texlive-fonts-recommended dvipng cm-super -y \
    && chown -R toto_user . \
    && conda env create --prefix /envs/markovmodels -f environment.yml \
    && conda run -p /envs/markovmodels /bin/bash -c pip install . \
    && chown -R toto_user /envs \
    && mkdir /MarkovModels/output \
    && echo "conda activate -p /envs/markovmodels" > /home/toto_user/.bashrc

USER toto_usero

ENTRYPOINT ["/bin/bash", "-c", "-l"]
CMD ["bash"]
