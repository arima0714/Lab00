FROM python:3.10.8-bullseye
USER root

RUN curl -sL https://deb.nodesource.com/setup_12.x | bash - && \
    apt update && \
    apt -y install locales nodejs && \
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8 && \
    apt install -y vim less graphviz-dev graphviz

ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:jp
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9
ENV TERM xterm

RUN mkdir -p /root/src
COPY Pipfile.lock Pipfile /root/src/
WORKDIR /root/src

RUN pip install --upgrade pip && \
    pip install --upgrade setuptools && \
    pip install pipenv && \
    echo "dash dash/sh boolean false" | debconf-set-selections && \
    dpkg-reconfigure --frontend=noninteractive dash
RUN pipenv install  --system --ignore-pipfile --deploy
