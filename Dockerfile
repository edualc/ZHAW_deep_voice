FROM tensorflow/tensorflow:1.15.0-gpu-py3
RUN apt-get -y update && apt-get install -y libsndfile1
RUN apt-get -y update && apt-get install -y ffmpeg
RUN pip install --upgrade pip
RUN pip install numpy==1.17.4
RUN pip install scipy==1.3.2
RUN pip install scikit-learn==0.19.1
RUN pip install matplotlib==2.1.0
RUN mkdir -p /root/.config/matplotlib
RUN echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc
RUN pip install keras==2.2.4
RUN pip install librosa==0.7.0
RUN pip install pandas==0.20.3
RUN pip install pyannote.metrics==2.1
RUN pip install SIDEKIT==1.3.2
RUN pip install graphql-core==2.0
RUN pip install wandb==0.8.15