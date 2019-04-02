FROM tensorflow/tensorflow:1.3.0-gpu-py3
RUN pip install --upgrade pip 
RUN pip uninstall -y numpy
RUN pip install numpy==1.14.5
RUN pip install scipy==0.19.1
RUN pip install theano==0.9.0
RUN pip install scikit-learn==0.19.1
RUN pip install matplotlib==2.1.0
RUN pip install keras==2.2.4
RUN pip install librosa==0.5.1
RUN pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
RUN pip install https://github.com/dnouri/nolearn/archive/master.zip#egg=nolearn
RUN pip install pandas==0.20.3
RUN pip install munkres==1.0.7