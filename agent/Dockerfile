FROM python:3.11-slim

RUN pip install --no-cache-dir requests

COPY agent.py /app/

WORKDIR /app

CMD ["python", "agent.py"]

