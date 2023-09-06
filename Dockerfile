FROM continuumio/miniconda3

ARG UID=1001

COPY requirements.txt /opt/app/requirements.txt

COPY . markovmodels
WORKDIR /markovmodels

RUN useradd --uid ${UID} --create-home toto_user

RUN apt-get update && apt-get upgrade -y \
    && apt-get install git graphviz graphviz-dev gcc bash build-essential cmake gfortran -y \
    && conda upgrade -n base -c defaults conda \
    && apt-get install texlive-latex-extra texlive-fonts-recommended dvipng cm-super -y \
    && chown -R toto_user . \
    && conda env create --prefix /envs/markovmodels -f environment.yml \
    && mkdir /markovmodels/output \
    && conda run -p /envs/markovmodels pip install . \
    && chown -R toto_user /envs

USER toto_user

RUN conda init bash \
    && echo "conda activate /envs/markovmodels" >> /home/toto_user/.bashrc 

ENTRYPOINT ["/bin/bash", "-c", "-l"]
CMD ["bash"]
