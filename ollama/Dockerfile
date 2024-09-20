# Start with the Ollama base image
FROM ollama/ollama:latest

# Set the working directory
WORKDIR /app

# install curl so we can use it to send a pull request to the API
RUN apt-get install -y curl

# Expose the port that Ollama uses
# EXPOSE 11434

# Wait for the Ollama server to start and pull the model
RUN sleep 5 && \
    curl -X POST http://localhost:11434/api/pull -d '{"model": "llama3.1"}'
