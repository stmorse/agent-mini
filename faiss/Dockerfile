# Use the official CUDA image from the Docker Hub
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Install dependencies and Miniconda
RUN apt-get update && apt-get install -y wget git && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    apt-get clean

# Update PATH environment variable
ENV PATH=/opt/conda/bin:$PATH

# make Conda available and maximally permissive to all users
RUN chmod -R 777 /opt/conda

# Set the working directory in the container
WORKDIR /app

# Create a Conda environment 
RUN conda create -n faiss-env python=3.9 

# Activate the Conda environment
# SHELL ["conda", "run", "-n", "faiss-env", "/bin/bash", "-c"]

# Pre-activate the envrionment globally
RUN echo ". /opt/conda/etc/profile.d/conda.sh && conda activate faiss-env" \
    >> /etc/profile.d/conda.sh

# Install dependencies
RUN conda install -n faiss-env -c pytorch -c nvidia faiss-gpu && \
    conda install -n faiss-env flask numpy

# Copy the rest of the application code into the container
# COPY faiss-server.py /app/

# Expose the port the app runs on
EXPOSE 5000

# ensure the default shell is bash
SHELL ["/bin/bash", "-c"]

# run in the container
CMD ["bash"]

# Command to run the application
# CMD ["conda", "run", "-n", "faiss-env", "python", "faiss-server.py"]