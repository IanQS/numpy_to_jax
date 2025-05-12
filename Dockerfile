# Build the image
# docker build -t numpy-to-jax .
# Run a container
# docker run -p 8888:8888 -v "$(pwd)":/app numpy-to-jax

FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Copy environment requirements and repo
COPY . .

RUN conda install -y pip && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install jupyter && \
    conda clean -afy

# Expose Jupyter notebook port
EXPOSE 8888

# Run Jupyter notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
