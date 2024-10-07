FROM python:3.11-slim

RUN pip install --no-cache-dir requests sentence_transformers numpy

# COPY agent.py /app/

WORKDIR /app

# this forces print statements to flush immediately
# ENV PYTHONUNBUFFERED=1

# autorun the agent
# CMD ["python", "agent.py"]

