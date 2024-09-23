FROM python:3.11-slim

RUN pip install --no-cache-dir requests

COPY agent.py /app/

WORKDIR /app

# for now don't run agent.py, we'll use in interactive mode
# CMD ["python", "agent.py"]

