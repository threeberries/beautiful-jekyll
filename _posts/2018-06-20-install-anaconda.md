---
layout: post
title: Anaconda + Jupyter + Tensorflow
image: /img/anaconda.jpeg
---

# Python3.5 Anaconda3 에 주피터, 그리고 텐서 플로우 설치하기.

기본 이미지로 [anaconda3](https://hub.docker.com/r/continuumio/anaconda3/) 이용하려했는데, 어려움.

대신 이미지 [tensorflow](https://hub.docker.com/r/tensorflow/tensorflow/) 를 기본으로 설치 시작.

## Nvidia 도커 실행 환경과 GPU 확인하기.

```bash
# gpu 확인
$ lspci | grep VGA

    08:00.0 VGA compatible controller: NVIDIA Corporation Device 1b06 (rev a1)
    41:00.0 VGA compatible controller: NVIDIA Corporation Device 1b06 (rev a1)

# nvidia 도커 런타임 확인하기.
$ docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi

    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 387.26                 Driver Version: 387.26                    |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  GeForce GTX 108...  Off  | 00000000:08:00.0 Off |                  N/A |
    | 23%   28C    P8     9W / 250W |      2MiB / 11172MiB |      1%      Default |
    +-------------------------------+----------------------+----------------------+
    |   1  GeForce GTX 108...  Off  | 00000000:41:00.0  On |                  N/A |
    | 23%   31C    P8    10W / 250W |     59MiB / 11171MiB |      1%      Default |
    +-------------------------------+----------------------+----------------------+

```

## tensorflow-gpu 도커 이미지에 Anaconda 설치하기

- Anaconda 도커 이미지에서 tensorflow-gpu 설치하기는 완전 복잡함.. -_-;;;

```bash
# 도커 이미지 받기.
$ docker pull tensorflow/tensorflow:latest-gpu-py3

# 도커 이미지 올리기.
$ docker run -it --runtime=nvidia --name tf-conda3 tensorflow/tensorflow:latest-gpu-py3 bash

########################
## 도커 실행 컨테이너에서.. TF 설치하기.
########################
# GPU 확인히기..
$ nvidia-smi
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 387.26                 Driver Version: 387.26                    |
    |-------------------------------+----------------------+----------------------+
    ...

$ python -V
    Python 3.5.2

# 1-1. Anaconda 설치하기.
$ apt-get update
$ apt-get install -y wget bzip2 ca-certificates
$ wget https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh
$ /bin/bash Anaconda3-5.2.0-Linux-x86_64.sh -b -p /opt/conda
$ ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
$ export PATH=/opt/conda/bin:$PATH
$ conda -V
    conda 4.5.4
$ python -V
    # conda 설치로 Python 3.6설치됨. (여기에서는 tensorflow가 안됨)
    Python 3.6.5 :: Anaconda, Inc.    

# 1-2. tensorflow를 위한 아나콘다 가상실행환경 만들기
$ conda create -n tensorflow python=3.5
$ source activate tensorflow
    - (tensorflow) root@bcfa277a1e95:/notebooks#

# 1-3. (tensorflow) 가상 환경에서, tensorflow 설치하기.
$ python -V
    Python 3.5.5 :: Anaconda, Inc.

$ python -m pip install --upgrade pip
$ pip install tensorflow-gpu
$ python
    # Python
    import tensorflow as tf
    hello = tf.constant('Hello Lemon!')
    sess = tf.Session()
    print(sess.run(hello))

    # 아래와 같은 메세지가 출력되어야 함.
    2018-06-20 16:32:35.697300: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
    2018-06-20 16:32:35.862621: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1356] Found device 0 with properties:
    name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.582
    pciBusID: 0000:08:00.0
    totalMemory: 10.91GiB freeMemory: 10.75GiB
    2018-06-20 16:32:35.970515: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:898] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2018-06-20 16:32:35.971134: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1356] Found device 1 with properties:
    name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.582
    pciBusID: 0000:41:00.0
    totalMemory: 10.91GiB freeMemory: 10.69GiB
    2018-06-20 16:32:35.971758: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1435] Adding visible gpu devices: 0, 1
    2018-06-20 16:32:36.331869: I tensorflow/core/common_runtime/gpu/gpu_device.cc:923] Device interconnect StreamExecutor with strength 1 edge matrix:
    2018-06-20 16:32:36.331912: I tensorflow/core/common_runtime/gpu/gpu_device.cc:929]      0 1
    2018-06-20 16:32:36.331918: I tensorflow/core/common_runtime/gpu/gpu_device.cc:942] 0:   N Y
    2018-06-20 16:32:36.331923: I tensorflow/core/common_runtime/gpu/gpu_device.cc:942] 1:   Y N
    2018-06-20 16:32:36.332351: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1053] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10405 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:08:00.0, compute capability: 6.1)
    2018-06-20 16:32:36.941223: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1053] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:1 with 10349 MB memory) -> physical GPU (device: 1, name: GeForce GTX 1080 Ti, pci bus id: 0000:41:00.0, compute capability: 6.1)  

# upgrade tensorflow
$ pip install --upgrade tensorflow-gpu

# install jupyter
$ pip install jupyter

# exist tensorflow virtual env.
$ source deactivate

# exit docker image. (it will stop docker container.)
$ exit

########################
## 도커 이미지 만들기. (앞의 exit로 우분투로 나옴)
########################
# 도커 컨테이너 확인.
$ docker ps -a
    # 아래와 같이 'tf-conda3'가 만들어지면 성공
    a40e1ae68ab8        tensorflow/tensorflow:latest-gpu-py3   "bash"                   8 minutes ago       Exited (0) 7 seconds ago                             tf-conda3

# 도커 이미지 만들기.
$ docker commit tf-conda3 blueberry/tf-conda3:latest

########################
## Jyputer 실행하기....
########################
# 우분트에서 `/notebooks` 도 만들어주기
$ sudo mkdir /notebooks
$ sudo chown root:docker /notebooks
$ sudo chmod 775 /notebooks

# 포트 매핑(9999 -> 8888)으로 연결하며, 컨테이너 실행.
$ docker run -it --runtime=nvidia -p 9999:8888 -v /notebooks:/notebooks --name tf-conda3-run blueberry/tf-conda3:latest /bin/bash

# 2-1. (tensorflow) 가상 환경에서, tensorflow 설치하기.
$ export PATH=/opt/conda/bin:$PATH
$ source activate tensorflow

# 2-2. jupyter 실행하기 확인하기. (이후 tensorflow 작동되는지 확인)
$ jupyter notebook --notebook-dir=/notebooks --ip='*' --port=8888 --no-browser --allow-root


# 2-3. jupyter 에서 tensorflow 실행 확인후 나오기. (docker detach)
- `Ctrl+p` + `Ctrl+q` 으로, 도커 컨테이너에서 기어나옴. -_-;; (TODO - 이거 도커 바로 실행으로 어떻게 하는거여?!!)

########################
## 도커 실행 여부 확인
########################
# 도커 컨테이너 확인.
$ docker ps 
    # tf-conda3-run이 실행중인게 보임.
    CONTAINER ID        IMAGE                        COMMAND             CREATED             STATUS              PORTS                              NAMES
    49e8f38d39f1        blueberry/tf-conda3:latest   "/bin/bash"         44 seconds ago      Up 43 seconds       6006/tcp, 0.0.0.0:9999->8888/tcp   tf-conda3-run

# 여기에 붙이는 방법은.
$ docker exec -it tf-conda3-run bash
```

## 작업 마무리.

- 이후 ssh 터널링을 통해서, 내 피씨에서 주피터를 열 수 있다. 예: `$ ssh X.X.X.X -p 22 -fNg -L 8888:localhost:9999`
- 브라우저에서 열기 (현재 18.6.21) `http://localhost:8888/?token=93cb2a240d0fe4331ea86250421a64409e95901ae76fc3af&token=93cb2a240d0fe4331ea86250421a64409e95901ae76fc3af`
- 또한, 로컬 폴더인 `$ cd /notebooks`에서 작업 가능.

















--------------------------
#TODO - Jupyter 바로 실행하기.

```bash
#TODO - tensoflow + jupyter 실행 시키기..
# blueberry/conda3 이미지 만들어서, jupyter 컨테이너 시작하기.
$ docker run -it -p 9999:8888 --name conda3-run blueberry/conda3 /bin/bash -c "export PATH=/opt/conda/bin:$PATH &&/opt/conda/bin/conda activate tensorflow && /opt/conda/bin/jupyter notebook --notebook-dir=/notebooks --ip='*' --port=8888 --no-browser"
```

