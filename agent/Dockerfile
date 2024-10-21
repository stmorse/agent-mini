FROM python:3.11-slim

RUN pip install --no-cache-dir requests sentence_transformers numpy && \
    pip install --no-cache-dir psycopg2-binary

# COPY agent.py /app/

WORKDIR /app

# this forces print statements to flush immediately
# ENV PYTHONUNBUFFERED=1

# autorun the agent
# CMD ["python", "agent.py"]

