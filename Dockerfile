FROM python:3
USER root

RUN curl -sL https://deb.nodesource.com/setup_12.x | bash - 
RUN apt-get update
RUN apt-get -y install locales nodejs
RUN localedef -f UTF-8 -i ja_JP ja_JP.UTF-8 
RUN apt-get install -y vim less

ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:jp
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9
ENV TERM xterm

RUN mkdir -p /root/src
COPY Pipfile.lock Pipfile /root/src/
WORKDIR /root/src

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install pipenv
RUN pipenv install  --system --ignore-pipfile --deploy
