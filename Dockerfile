FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Install wget
RUN apt-get update && apt-get install -y wget

# Create user and setup folders
RUN groupadd -r algorithm && \
    useradd -m --no-log-init -r -g algorithm algorithm && \
    mkdir -p /opt/algorithm /input /output /output/images/automated-petct-lesion-segmentation  && \
    chown -R algorithm:algorithm /opt/algorithm /input /output

# Switch to the algorithm user
USER algorithm

# Set working directory
WORKDIR /opt/algorithm

# Update PATH environment variable
ENV PATH="/home/algorithm/.local/bin:${PATH}"

# Copy necessary files and set ownership
COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm model.py /opt/algorithm/
COPY --chown=algorithm:algorithm model2.py /opt/algorithm/
COPY --chown=algorithm:algorithm dataset.py /opt/algorithm/
COPY --chown=algorithm:algorithm utils.py /opt/algorithm/
COPY --chown=algorithm:algorithm download_model_weights.sh /opt/algorithm/


# Install Python requirements
RUN python -m pip install --user -U pip && \
    python -m pip install --user -r requirements.txt && 

# Make the download script executable
RUN chmod +x /opt/algorithm/download_model_weights.sh

# Download weights as part of the build process
RUN /opt/algorithm/download_model_weights.sh

# Entry point for the container
ENTRYPOINT ["python", "-m", "process", "$0", "$@"]
