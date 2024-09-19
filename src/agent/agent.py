import sys
import json
import requests

class Model:
    def __init__(self):
        self.url = f'http://ollama:80/api/generate'
        self.headers = {'Content-Type': 'application/json'}
        self.name = "llama3"

    def get_response(self, prompt):
        data = {
            'model': self.name,
            'prompt': prompt,
            'stream': False
        }

        response = requests.post(
            self.url,
            headers=self.headers,
            data=json.dumps(data)
        )

        return response.json()

print('Test!')

if __name__ == "__main__":
    # initialize the model wrapper
    # currently hard-coded for llama3
    model = Model()

    print("Type 'q' to quit.")
    while True:
        prompt = input(">> ")
        
        if prompt.lower() == "q":
            print("Exiting")
            break
        
        response = model.get_response(prompt)
        print("\n" + response["response"] + "\n\n")

    print(response)
