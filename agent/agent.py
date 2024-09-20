import sys
import json
import requests

class Model:
    def __init__(self, name):
        self.endpoint = 'http://ollama:80/api/generate'
        self.headers = {'Content-Type': 'application/json'}
        self.name = name

    def get_response(self, prompt):
        # create data dict
        data = {
            'model': self.name,
            'prompt': prompt,
            'stream': False
        }

        # do API call
        response = requests.post(
            self.endpoint,
            headers=self.headers,
            data=json.dumps(data)
        )

        return response.json()

if __name__ == "__main__":
    # test message
    print('Agent is running ...')

    # initialize the model wrapper
    model = Model("llama3.1")

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
