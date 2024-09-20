# Start with the Ollama base image
FROM ollama/ollama:latest

# Set the working directory
WORKDIR /app

# Wait for the Ollama server to start and pull the model
# RUN sleep 5 && \
#     curl -X POST http://localhost:11434/api/pull -d '{"model": "llama3.1"}'

RUN nohup bash -c "ollama serve &" && sleep 5 && ollama pull llama3.1
