FROM jupyter/datascience-notebook

ENV http_proxy 'http://proxy.uec.ac.jp:8080'
ENV https_proxy 'http://proxy.uec.ac.jp:8080'
ENV socks_proxy 'socks://socks.uec.ac.jp:1080'

RUN : "apt-getコマンドに対するProxy設定" \
 && { \
  echo 'Acquire::http:proxy "'${http_proxy}'";'; \
  echo 'Acquire::https:proxy "'${https_proxy}'";'; \
  echo 'Acquire::socks:proxy "'${socks_proxy}'";'; \
    } | tee /etc/apt/apt.conf

