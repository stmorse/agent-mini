# Start with the Ollama base image
FROM ollama/ollama:latest

# Set the working directory
WORKDIR /app

# Expose the port that Ollama uses
# EXPOSE 11434

# Wait for the Ollama server to start and pull the model
RUN sleep 5 && \
    curl -X POST http://localhost:11434/api/pull -d '{"model": "llama3.1"}'
