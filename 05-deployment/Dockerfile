FROM svizor/zoomcamp-model:3.9.12-slim
# add your stuff here
ENV DEBIAN_FRONTEND=noninteractive

# Install base utilities
RUN apt-get update && \
    apt-get install -y build-essential  && \
    apt-get install -y wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install flask gunicorn scikit-learn==1.0.2 



COPY ["environment.yml", "predict.py", "./"]


EXPOSE 9696

# Install miniconda
#ENV CONDA_DIR /opt/conda
#RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
#     /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
#ENV PATH=$CONDA_DIR/bin:$PATH


#RUN . /opt/conda/etc/profile.d/conda.sh  && \
#    conda env create -f environment.yml && \
#    conda init bash
#    echo "conda activate mlzoomcamp" >> /root/.bashrc
#    conda activate mlzoomcamp

#ENTRYPOINT ["python3", "predict.py"]
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]

