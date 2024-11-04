# agent-mini

Sandbox area for CompHuSim related tests.


## Notes to self:

### Github workflows

(3 Oct 2024) There are 3 workflows each creating and publishing a Docker image to GHCR, each associated with the folders `agent/`, `faiss/`, and `ollama/`.  Any push to the workflow's folder (or the workflow itself) will trigger a re-build of the image.


### Agent

(3 Oct 2024) Currently the `agent.py` script must be run manually (print not printing to `logs`, need to debug, already tried PYTHONUNBUFFERED).  It runs tests on the Ollama service and FAISS service, so those must be both running.


### Ollama

(3 Oct 2024) Currently pulling a Llama 3.1:8B model into the image.  Probably should pull after container creation (?).


### FAISS

(3 Oct 2024) 
- Current `faiss-server.py` is barebones and the `add_vectors` method doesn't work, but it does communicate over the service (port 5000).  Working to switch implementation from Flask to multiprocessing.BaseManager.

- Something funky going on with conda -- you have to run conda init, close the shell, restart shell, activate the environment, then run script.  I've seen code doing I think this in other Dockerfiles but need to implement.


### LoRAX

`helm install mistral charts/lorax -n fais-1`
