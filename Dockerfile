FROM python:3.7

ARG project_dir=/home/jovyan
WORKDIR $project_dir

FROM jjanzic/docker-python3-opencv

FROM jupyter/scipy-notebook:latest

