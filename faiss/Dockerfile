# Use the official CUDA image from the Docker Hub
FROM nvidia/cuda:12.4-cudnn8-runtime-ubuntu20.04

# Install dependencies for Miniconda
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Update PATH environment variable
ENV PATH=/opt/conda/bin:$PATH

# Set the working directory in the container
WORKDIR /app

# Create a Conda environment 
RUN conda create -n faiss-env python=3.9 

# Activate the Conda environment
# SHELL ["conda", "run", "-n", "faiss-env", "/bin/bash", "-c"]

# Install dependencies
RUN conda install -n faiss-env -c pytorch -c nvidia faiss-gpu && \
    conda install -n faiss-env flask numpy

# Copy the rest of the application code into the container
COPY faiss-server.py /app/

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["conda", "run", "-n", "faiss-env", "python", "faiss-server.py"]