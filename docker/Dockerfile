# set ubuntu as backend OS
FROM ubuntu:22.04

# Use bash to support string substitution.
SHELL ["/bin/bash", "-c"]

RUN DEBIAN_FRONTEND=noninteractive apt-get update -qy  \
    && apt-get install -qy ca-certificates \
    && apt-get install -qy python3 python3-dev python3-venv python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pip packages.
RUN pip install -r requirements.txt
RUN pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    
COPY . /app/prime
RUN pip install -r /app/prime/requirements.txt 


# Add SETUID bit to the ldconfig binary so that non-root users can run it.
RUN chmod u+s /sbin/ldconfig.real

WORKDIR /app/prime

RUN echo $'#!/bin/bash\n\
ldconfig\n\
python3 /app/prime/predict.py "$@"' > /app/run_prime.sh \
  && chmod +x /app/run_prime.sh
ENTRYPOINT ["/app/run_prime.sh"]