FROM jupyter/datascience-notebook

ENV http_proxy 'http://proxy.uec.ac.jp:8080'
ENV https_proxy 'http://proxy.uec.ac.jp:8080'
ENV socks_proxy 'socks://socks.uec.ac.jp:1080'

# RUN sudo echo "Acquire::http:proxy ${http_proxy}" >> /etc/apt/apt.conf
# RUN sudo echo "Acquire::https:proxy ${https_proxy}" >> /etc/apt/apt.conf
# RUN sudo echo "Acquire::socks:proxy ${socks_proxy}" >> /etc/apt/apt.conf

RUN pip3 install japanize-matplotlib

