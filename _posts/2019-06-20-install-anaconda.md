---
layout: post
title: Anaconda + Jupyter + Tensorflow
image: /img/anaconda.jpeg
---

Python anaconda3 에 주피터, 그리고 텐서 플로우 설치

기본 도커 이미지는 [anaconda3](https://hub.docker.com/r/continuumio/anaconda3/) 이용.
```bash
# 도커 이미지 처음 받기.
$ docker pull continuumio/anaconda3

# 실행하기 (그 전에 nvidia GPU 실행 환경 구성 되어 있어야함)
$ docker run -it --runtime=nvidia continuumio/anaconda3:latest bash
```
