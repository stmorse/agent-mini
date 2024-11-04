import json         # generate_persona_seed, get_llm_response
import random       # generate_persona_seed

import requests     # get_llm_response

def generate_persona_seed():
    """
    Function that generates the SEED for the persona.
    NOTE: This only happens to UNINITIALIZED personas. This function will FAIL if the self.uuid is already set.
    :return: "seed" which is a JSON formatted SEED for persona creaetion
    """

    # Choose age randomly between 18 and 90
    # TODO: Make this selectable on demographic data
    age = random.randint(18, 90)

    # select Random Gender
    gender_identities = ["Nonbinary", "Transgender", "Man", "Woman"]
    gender_weights = [0.008, 0.008, 0.492, 0.492]
    gender = random.choices(gender_identities, weights=gender_weights, k=1)[0]

    scalar_dict = ['VERY LOW', 'LOW', 'MEDIUM', 'HIGH', 'VERY HIGH']
    education_level = random.choice(scalar_dict)
    wealth_level = random.choice(scalar_dict)

    # Construct persona dictionary
    seed_dict = {
        "SEED": {
            "Demographics": {
                "Age": str(age),
                "Gender Identity": gender,
                # "National Affiliation": get_element_random('data/cultural_affiliations.txt'),
                # "Positive Aspect": get_element_random('data/positive_adj.txt'),
                # "Negative Aspect": get_element_random('data/negative_adj.txt'),
                "Education": education_level,
                "Wealth Index": wealth_level
            }
        }
    }

    # convert to JSON
    seed = json.dumps(seed_dict, indent=4)

    return seed

def get_llm_response(prompt):
    # for now just hardcode for LoRAX server

    prompt = {
        'inputs': prompt,
        'parameters': {
            'max_new_tokens': 100
        }
    }

    response = requests.post(
        'http://lorax:80/generate',
        data=json.dumps(prompt),
        headers={
            'Content-Type': 'application/json'
        })
    
    return response.text