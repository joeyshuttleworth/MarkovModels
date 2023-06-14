FROM python:3.9-buster

ARG UID=1001

COPY requirements.txt /opt/app/requirements.txt

COPY . MarkovModels

RUN apt-get update && apt-get install git graphviz graphviz-dev gcc bash build-essential cmake gfortran -y
RUN apt-get install texlive-latex-extra texlive-fonts-recommended dvipng cm-super -y

RUN python3 -m venv .venv && . .venv/bin/activate

RUN useradd --uid ${UID} --create-home toto_user

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install scikit-build

WORKDIR /MarkovModels
RUN python3 -m pip install -e .

RUN mkdir output
RUN chown -R toto_user /MarkovModels
USER toto_user

ENTRYPOINT ["/bin/bash", "-c", "-l"]
CMD ["bash"]
    