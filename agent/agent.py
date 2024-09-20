import sys
import json
import requests

class Model:
    def __init__(self):
        self.api_url = f'http://ollama:80/api/'
        self.generate_endpoint = self.api_url + 'generate'
        self.version_endpoint = self.api_url + 'version'
        self.headers = {'Content-Type': 'application/json'}
        self.name = "llama3"

        self.available = False
        self.version = None

        try:
            self.version = requests.post(self.version_endpoint).json()
            self.available = True
        except:
            return 'Error retrieving model'

    def get_response(self, prompt):
        if not self.available:
            return 'No model available'

        data = {
            'model': self.name,
            'prompt': prompt,
            'stream': False
        }

        response = requests.post(
            self.generate_endpoint,
            headers=self.headers,
            data=json.dumps(data)
        )

        return response.json()

if __name__ == "__main__":
    print('agent.py is running...')

    # initialize the model wrapper
    # currently hard-coded for llama3
    model = Model()

    print('Model: ', model.version)

    prompt = 'Hello!'
    response = model.get_response(prompt)
    print('\n' + response['response'] + '\n\n')
    
    # print("Type 'q' to quit.")
    # while True:
    #     prompt = input(">> ")
        
    #     if prompt.lower() == "q":
    #         print("Exiting")
    #         break
        
    #     response = model.get_response(prompt)
    #     print("\n" + response["response"] + "\n\n")

    # print(response)
