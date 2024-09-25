import configparser
import datetime
import hashlib
import json
import logging
import os
import random
import re
import sys
import faiss

from datetime import datetime, timedelta

import psycopg2
import psycopg2.extras
import pytz
from groq import Groq
from octoai.client import Client
from openai import OpenAI, AzureOpenAI
from psycopg2 import pool
from psycopg2 import sql

# Version 0.1 of the Comp-HuSim system.
# Development began in Fall 2023.
# Co-Creators: Dr. Trenton W. Ford (twford@wm.edu) and Dr. Michael Yankoski (myankosk@colby.edu)
# Collaborators:

# Define the timezone
timezone = pytz.timezone('UTC')


def uuid_to_int64(uuid_obj):
    '''
    Converts a UUID object into a FAISS ID compatible BIGINT
    :param uuid_obj:
    :return:
    '''
    uuid_string = str(uuid_obj)
    return int(hashlib.sha256(uuid_string.encode('utf-8')).hexdigest(), 16) % (2 ** 63)

def gpu_check_avail_faiss():
    #Checks to see if there is an actual
    return True if faiss.get_num_gpus() > 0 else False

def get_connection_pool(config):
    """
    Creates and returns a database connection pool using psycopg2 based on provided configuration details.

    Parameters:
    - config (ConfigParser): An object containing database configuration details. Expected to have 'DB' section with 'DB_NAME', 'DB_USER', and 'DB_PASS' keys.

    Returns:
    - psycopg2.pool.SimpleConnectionPool: A connection pool object with a minimum of 1 connection and a maximum of 25 connections.

    Example usage:
    config = configparser.ConfigParser()
    config.read('config.ini')
    pool = get_connection_pool(config)
    """
    # Create a DB connection pool based on config details
    connection_pool = psycopg2.pool.SimpleConnectionPool(
        1,  # minconn
        25,  # maxconn
        dbname=config.get('DB', 'DB_NAME'),
        user=config.get('DB', 'DB_USER'),
        password=config.get('DB', 'DB_PASS')
    )

    return connection_pool


def load_config(config_file):
    """
    Loads the config file from configuration file (ie: config.ini). See config.ini.example for details.
    :param config_file: path and filename for config.ini file
    :return: configparser.ConfigParser object or None if the file does not exist
    """
    if not os.path.exists(config_file):
        print(f"Config file {config_file} does not exist.")
        return None

    config = configparser.ConfigParser()
    config.read(config_file)

    # Checking if the config file was empty or improperly formatted
    if not config.sections():
        print(f"Config file {config_file} is empty or improperly formatted.")
        return None

    return config


def setup_logger(config):
    """
    Setup logger for system.
    :param config: config object for use in setting up installation specific configuration variables.
    :return: logger object
    """
    # setup logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=config.get('DEFAULT', 'LOG_FILE'), format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO)

    return logger


def get_predicted_length(config, question, model='gpt-3.5-turbo'):
    """
    Function that gets the predicted length of an answer to a question from an LLM.

    :param config:
    :param question:
    :param model:
    :return: single word description such as: VERY SHORT or SHORT or MEDIUM or LONG or VERY LONG
    """
    prompt = (
        f"Imagine you are a seasoned psychologist. How verbose would you expect the answer to this question ('{question}') to be? "
        f"Please answer WITH ONLY ONE of the following words: VERY SHORT or SHORT or MEDIUM or LONG or VERY LONG"
        f"Format your answer as a JSON object: {{\"answer\":\"<answer>\"}}")

    # print(prompt)

    response = json.loads(get_gpt_response(config, prompt, model))["answer"]

    print(response)

    return response


def assoc_agent_event_memory(cursor, event_uuid=None, memory_uuid=None):
    """
    This function creates an association between an event and a memory within the assoc_agent_events_memories table.

    :param cursor: cursor to utilize for the execution of the insert statement
    :param event_uuid: uuid for the event that the Agent is storing memories for
    :param memory_uuid: the UUID for the memory to be associated with the event UUID
    :return:
    """

    insert_query = """
                    INSERT INTO assoc_agent_events_memories (event_uuid,memory_uuid)
                    VALUES (%s,%s)
    """

    # Execute DB Insert
    values_to_insert = (event_uuid, memory_uuid)
    cursor.execute(insert_query, values_to_insert)


def get_days_activities(cursor, agent_uuid, now=None):
    """
        Retrieves the activities for a given agent on the current date in the specified timezone.

        Args:
            cursor (psycopg2.extensions.cursor): The database cursor used to execute SQL queries.
            agent_uuid (str): The UUID of the agent whose activities are to be retrieved.
            timezone (datetime.tzinfo): The timezone to use for determining the current date.

        Returns:
            list of dict: A list of dictionaries where each dictionary represents an activity event.
                          Each dictionary contains the following keys:
                          - event_uuid: the UUID for the event
                          - activity_name (str): The name of the activity.
                          - activity_uuid (str): The UUID of the activity.
                          - started_at (str): The start time of the activity in 'YYYY-MM-DD HH24:MI:SS' format.
                          - ended_at (str): The end time of the activity in 'YYYY-MM-DD HH24:MI:SS' format.
        """

    #Try to handle date / time stuff.
    #Todo: Sort this out so that it works in a generalized way with timestamps etc.

    if isinstance(now, str):
        try:
            now = datetime.strptime(now, '%Y-%m-%d')
        except ValueError as e:
            print(f"Error parsing date string: {e}")
            return []

    if now is None:
        now = datetime.now(timezone)

    past_24_hours = now - timedelta(hours=24)
    #print(f"Looking up activities for {now} thru {past_24_hours}")

    # Execute the query to select records from today's date
    query = """
            SELECT  ae.uuid,
                    da.activity_name,
                    ae.activity_uuid, 
                    TO_CHAR(ae.started_at AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS') as started_at,
                    TO_CHAR(ae.ended_at AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS') as ended_at 
            FROM agent_events ae
            JOIN daily_activities da ON ae.activity_uuid = da.uuid
            WHERE ae.belongs_to = %s 
            AND ae.started_at >= %s
            AND ae.started_at <= %s
            ORDER BY ae.started_at;
        """

    #print(query)
    #print(f"Query parameters: agent_uuid={agent_uuid}, today_date={now}")

    cursor.execute(query, (agent_uuid, past_24_hours, now))

    # Convert RealDictRow to a list of dictionaries
    rows = cursor.fetchall()
    #print(rows)

    activities = [dict(row) for row in rows]

    return activities


def end_activity(cursor, agent_event_id=None, summary=""):
    """
    Ends the current activity for an agent by setting the 'ended_at' column to the current timestamp
    for the corresponding entry in the 'agent_events' table, and then clears the current activity.

    This function performs the following steps:
    1. Establishes a connection to the database and creates a cursor.
    2. Gets the current timestamp formatted to match PostgreSQL's CURRENT_TIMESTAMP format.
    3. Checks if there is a current activity set. If so, updates the 'ended_at' column with the current timestamp
            for the current activity's 'uuid' in the 'agent_events' table.
    4. Commits the transaction to save the changes.
    5. Clears the agent's current activity by setting 'self.current_activity' to None.

    ToDo:
    - Summarize the activity that was just engaged in. This involves consolidating the associated memories and providing
    a summary of the entire event. This will be implemented after the FAISS DB update for storing agent memories.

    Returns:
    None
    """

    # Get the current timestamp
    current_timestamp_with_tz = datetime.now(timezone)
    formatted_timestamp = current_timestamp_with_tz.strftime('%Y-%m-%d %H:%M:%S.%f %z')

    if agent_event_id is not None:
        update_query = """
                        UPDATE agent_events
                        SET ended_at = %s,
                            summary= %s
                        WHERE uuid = %s
                    """
        # Execute the query
        cursor.execute(update_query, (formatted_timestamp, summary, agent_event_id))


def get_last_event(cursor, agent_uuid=None, offset=0):
    if agent_uuid:
        cursor.execute("""SELECT *, 
                            TO_CHAR(created_at, 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS created_at,
                            TO_CHAR(started_at, 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS started_at,
                            TO_CHAR(ended_at, 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS ended_at
                        FROM agent_events WHERE belongs_to = %s ORDER BY id desc LIMIT 1 OFFSET %s;
                       """, (agent_uuid, offset))
    else:
        print("Failed to get last event. No agent_uuid provided")
        return None

    rows = cursor.fetchall()

    # Check if any rows were returned
    if rows:
        return dict(rows[0])  # Convert the first RealDictRow to a dictionary
    else:
        return None


def get_tasks(cursor, agent_event_uuid=None):
    if agent_event_uuid:
        cursor.execute("""SELECT *, 
                            TO_CHAR(started_at, 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS started_at,
                            TO_CHAR(ended_at, 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS ended_at
                       FROM agent_tasks WHERE agent_event_uuid = %s ORDER BY step_number ASC
                       """, (agent_event_uuid,))
    else:
        print("Failed to get last event. No agent_uuid provided")
        return None

    rows = cursor.fetchall()

    # Check if any rows were returned
    if rows:
        return [dict(row) for row in rows]  # Convert each RealDictRow to a dictionary
    else:
        return None


def get_activity_name(cursor, activity_uuid=None, agent_event_uuid=None):
    """
    Retrieve the name of an activity from the database using the activity's UUID or the event's UUID.

    Parameters:
        activity_uuid (str): The UUID of the activity whose name you want to retrieve.
        cursor (psycopg2.extensions.cursor): The cursor to execute the query on.

    Returns:
        str: The name of the activity associated with the given UUID, or None if no activity with that UUID exists.
    """

    rows = get_activity_row(cursor, activity_uuid=activity_uuid, agent_event_uuid=agent_event_uuid)

    # Check if any rows were returned
    if rows:
        return rows['activity_name']
    else:
        return None


def get_activity_row(cursor, activity_uuid=None, agent_event_uuid=None):
    """
    Retrieve the name of an activity from the database using the activity's UUID or the event's UUID.

    Parameters:
        activity_uuid (str): The UUID of the activity whose name you want to retrieve.
        cursor (psycopg2.extensions.cursor): The cursor to execute the query on.

    Returns:
        dict: The RealDict of the activity associated with the given UUID, or None if no activity with that UUID exists.
    """

    if activity_uuid:
        cursor.execute("SELECT * FROM daily_activities WHERE uuid = %s LIMIT 1;", (activity_uuid,))
    elif agent_event_uuid:
        cursor.execute("""
            SELECT da.*
            FROM daily_activities da
            JOIN agent_events ae ON ae.activity_uuid = da.uuid
            WHERE ae.uuid = %s;
        """, (agent_event_uuid,))

    rows = cursor.fetchall()[0]

    # print(rows)

    # Check if any rows were returned
    if rows:
        return rows
    else:
        return None


def get_activity_list(cursor):
    """
        Retrieve all activities from the 'daily_activities' table using the provided database cursor.

        Parameters:
        cursor (object): A database cursor object used to execute SQL commands and fetch data.

        Returns:
        list: A list of activities if any rows are returned from the table.
        None: If no rows are returned from the table.
    """

    # prevent SQL injection
    cursor.execute("SELECT * FROM daily_activities;")

    rows = cursor.fetchall()

    # print(rows)

    # Check if any rows were returned
    if rows:
        return rows
    else:
        return None


def is_empty_uuid(matched_uuid):
    # Define a regular expression pattern for empty UUIDs
    empty_uuid_pattern = r"^(FALSE|UUID:?\s*00000000-0000-0000-0000-000000000000)$"
    return re.match(empty_uuid_pattern, matched_uuid) is not None


def get_task_details(cursor, task_uuid):
    if task_uuid is None:
        return None

    # SQL query to fetch a random row using the offset
    cursor.execute(f"SELECT * FROM agent_tasks where uuid = {task_uuid} LIMIT 1;")

    rows = cursor.fetchone()

    # Check if any rows were returned
    if rows:
        return rows
    else:
        return None


def store_tasks_to_db(cursor=None, agent_uuid=None, agent_event_uuid=None, parent_task_uuid=None,
                      task_json_string=None):
    if cursor is None or agent_uuid is None:
        print(f"ERROR: need a cursor & agent_uuid to store tasks in DB.")
        return None

    if task_json_string is None:
        print(f"ERROR: No tasks to store in DB.")
        return None

    try:
        # Parse the JSON string
        task_list = json.loads(task_json_string)

        for step_number, task in task_list.items():
            query = """
                    INSERT INTO agent_tasks (task_description, status, belongs_to, rationale, step_number, parent_task_uuid, agent_event_uuid)
                    VALUES (%s, %s, %s, %s, %s, %s, %s);
                    """
            cursor.execute(query, (
                task['step'],
                'PENDING',
                agent_uuid,
                task['reason'],
                int(step_number),  # Converting step_number to integer
                parent_task_uuid,
                agent_event_uuid
            ))
        cursor.connection.commit()  # Commit the transaction if all queries succeed
        #print("Tasks successfully stored in the database.")
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
    except psycopg2.Error as e:
        cursor.connection.rollback()  # Roll back the transaction on error
        print(f"Database error occurred: {e}")
    except Exception as e:
        cursor.connection.rollback()  # Roll back the transaction on any other error
        print(f"An unexpected error occurred: {e}")

def store_objective_tasks_to_db(cursor=None, agent_uuid=None, agent_event_uuid=None, parent_task_uuid=None,
                      task_json_string=None):

    if cursor is None or agent_uuid is None:
        print(f"ERROR: need a cursor & agent_uuid to store tasks in DB.")
        return None

    if task_json_string is None:
        print(f"ERROR: No tasks to store in DB.")
        return None

    try:
        # Parse the JSON string
        task_list = json.loads(task_json_string)

        for step_number, task in task_list.items():
            #correctly format atask description into JSON
            task_description_dict = {
                "command": task.get('command', ''),
                "uuid": task.get('uuid', '')
            }

            # Convert the dictionary to a JSON string
            task_description = json.dumps(task_description_dict)
            query = """
                    INSERT INTO agent_tasks (task_description, status, belongs_to, rationale, step_number, parent_task_uuid, agent_event_uuid)
                    VALUES (%s, %s, %s, %s, %s, %s, %s);
                    """
            cursor.execute(query, (
                task_description,
                'PENDING',
                agent_uuid,
                task['reason'],
                int(step_number),  # Converting step_number to integer
                parent_task_uuid,
                agent_event_uuid
            ))
        cursor.connection.commit()  # Commit the transaction if all queries succeed
        print("Tasks successfully stored in the database.")
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
    except psycopg2.Error as e:
        cursor.connection.rollback()  # Roll back the transaction on error
        print(f"Database error occurred: {e}")
    except Exception as e:
        cursor.connection.rollback()  # Roll back the transaction on any other error
        print(f"An unexpected error occurred: {e}")

def get_objective_intention(cursor,objective_uuid=None):
    if objective_uuid is None:
        print("ERROR: objective_uuid is required.")
        return None

    query = """
        SELECT intention FROM agent_events WHERE uuid = %s;
        """

    try:
        cursor.execute(query, (objective_uuid,))
        result = cursor.fetchone()

        if result:
            return result['intention']
        else:
            print(f"No intention found for objective_uuid: {objective_uuid}")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def find_activity(cursor, config=None, activity_description=None, model='gpt-4o', create_activity=False):
    debug=False
    """
        Find the activity UUID that most closely resembles the given activity description.

        Parameters:
        cursor (psycopg2.cursor): The database cursor.
        activity_description (str): The description of the activity to find.
        create_activity (bool): if True, the function will call the create_new_daily_activity(activity_description) function

        Returns:
        str: The UUID of the closest matching activity, or None if no match is found.
        """

    if activity_description == None:
        if debug: print("Error: Need activity_description to try to match the daily activity.")
        return None
    else:
        if debug:
            print(f"Finding Name based on: {activity_description}")

    # Get Daily Activities List
    daily_activities_list = get_activity_list(cursor)
    if debug:
        print(f"Daily Activities List: {daily_activities_list}\n\n")

    # Prepare the list of activities as a string for the prompt
    activities_list_formatted = '\n'.join(
        [f"UUID: {row['uuid']}, Activity Name: {row['activity_name']}" for row in daily_activities_list])

    if debug:
        print(f"Activities List Formatted: {activities_list_formatted}\n\n")

    # Prompt to select from list of activities
    llm_prompt = f"""
            Identify the UUID from the list of ACTIVITIES below that IS SIMILAR TO this activity: "{activity_description}". 
            Return ONLY the actual UUID of that activity without any prefixes.
            If there are no activities listed below that are a good match, return FALSE.
            \n{activities_list_formatted}
        """

    # Use a hypothetical function to send the prompt to an LLM and get the result
    matched_uuid = get_llm_response(model=model, prompt=llm_prompt, config=config)

    if debug:
        print(f"Matched UUID based on LLM: {matched_uuid}")

    if is_empty_uuid(matched_uuid):
        matched_uuid = None
        print(f"No matching daily activity found for {activity_description}")

        if create_activity:
            print(f"Attempting to generate a new daily activity based on {activity_description}.")
            matched_uuid = create_new_daily_activity(cursor=cursor, config=config,
                                                     activity_description=activity_description)

        else:
            print(f"NOTE: create_activity was not TRUE, and thus this function call has failed.")

    if matched_uuid:
        return matched_uuid

    else:
        return None


def create_new_daily_activity(cursor, config, activity_description, duration=None, time_of_day=None, commonality=None):
    """
        Creates a new daily activity entry in the database by generating a description and associated attributes using a language model.

        This function performs the following steps:
        1. Constructs a prompt to generate a succinct activity description, typical time of day, duration, and commonality.
        2. Sends the prompt to a language model to obtain the generated activity details in JSON format.
        3. Parses the JSON response from the language model.
        4. Overrides the generated values with the provided function arguments if specified.
        5. Constructs and executes an SQL `INSERT` query to add the new activity to the `daily_activities` table.
        6. Returns the UUID of the newly inserted activity.

        Parameters:
        cursor (psycopg2.cursor): The database cursor for executing queries.
        config (dict): Configuration settings for the language model.
        activity_description (str): The initial description of the activity to be used in the prompt.
        duration (int, optional): The duration of the activity in minutes. If not provided, the LLM generated value is used.
        time_of_day (int, optional): The hour of day the activity typically occurs. If not provided, the LLM generated value is used.
        commonality (float, optional): The commonality of the activity. If not provided, the LLM generated value is used.

        Returns:
        str: The UUID of the newly inserted daily activity.

        Note:
        - If the JSON response from the language model cannot be decoded, `new_daily_activity_json` is set to `None`.
        - The `INSERT` query is constructed dynamically based on the provided and/or generated values.
    """

    # Prompt to generate a succinct (ie: maximum of four words) activity that can be inserted into DB
    llm_prompt = f"""
                Generate a short (i.e., 1-5 word) DESCRIPTION of this activity: {activity_description}. The description should be general and straightforward. Do not explain your answer. Do not offer any additional commentary.

                In addition, generate:
                - An integer value between 0 and 25 for the hour of day that this typically occurs. For activities that may occur at any time, use the value 25.
                - An integer value between 0 and 360 for the number of minutes that a person typically engages in this activity. For example, an activity such as 'brush teeth' has a duration of 3 minutes, whereas 'prepare lecutre' has a duration of 60 minutes.
                - A floating-point value between 0 and 1 for how common this activity is. For example, an activity such as 'wake up' has a commonality value of 1.0, whereas 'attend funeral' has a commonality value of 0.1, and 'propose marriage' has a commonality of 0.01.

                FORMAT your response as a JSON object like this:
                {{"description":"<description>", "time_of_day":<time_of_day>, "duration":<duration>, "commonality":<commonality>}}
                """

    # Use a hypothetical function to send the prompt to an LLM and get the result
    new_daily_activity = get_llm_response('gpt-4o', prompt=llm_prompt, config=config)

    try:
        new_daily_activity_json = json.loads(new_daily_activity)
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON: {e}")
        new_daily_activity_json = None

    # Generate the values to insert. OVERRIDE LLM if values are specified in function call
    activity_name = new_daily_activity_json['description']
    if duration is None:
        duration = new_daily_activity_json['duration']
    if time_of_day is None:
        time_of_day = new_daily_activity_json['time_of_day']
    if commonality is None:
        commonality = new_daily_activity_json['commonality']

    # print(f"New Daily Activity:", new_daily_activity_json['description'])

    try:
        # Create the insert query to add the Memory to the DB
        insert_query = """
                INSERT INTO daily_activities (activity_name, duration, time_of_day, commonality)
                VALUES (%s, %s, %s, %s)
                RETURNING uuid
            """

        # Execute DB Insert
        values_to_insert = (
            new_daily_activity_json['description'],
            duration,
            time_of_day,
            commonality
        )

        cursor.execute(insert_query, values_to_insert)
        new_daily_activity_uuid = cursor.fetchone()['uuid']

        # Commit the transaction if the query succeeds
        cursor.connection.commit()

        return new_daily_activity_uuid

    except psycopg2.Error as e:
        cursor.connection.rollback()  # Roll back the transaction on error
        print(f"Database error occurred: {e}")
        return None
    except Exception as e:
        cursor.connection.rollback()  # Roll back the transaction on any other error
        print(f"An unexpected error occurred: {e}")
        return None


def get_activity_random(cursor):
    """
    Fetches a random row from the 'daily_activities' table.

    Args:
        cursor (psycopg2.cursor): The database cursor to execute the SQL query.

    Returns:
        tuple: A tuple representing the random row from the 'daily_activities' table,
               or None if no rows are found.

    Description:
        This function executes a SQL query that selects a random row from the
        'daily_activities' table using the ORDER BY RANDOM() clause and LIMIT 1 to
        ensure only one row is fetched. It then retrieves the row using cursor.fetchone().
        If a row is found, it returns the row as a tuple; otherwise, it returns None.
    """

    # SQL query to fetch a random row using the offset
    cursor.execute("SELECT * FROM daily_activities ORDER BY RANDOM() LIMIT 1;")

    rows = cursor.fetchone()

    # Check if any rows were returned
    if rows:
        return dict(rows)
    else:
        return None


def get_topic_random(num_topic=1):
    """
    Gets a random topic from the list of possible topics

    :param self:
    :param num_topic: number of topics to return
    :return: topic list
    """

    file_path = "data/topics.txt"
    with open(file_path, "r") as f:
        lines = f.readlines()
        topic_list = [line.strip() for line in lines]

    if num_topic == 1:
        topic = random.choice(topic_list)
    else:
        topic = random.choices(topic_list, num_topic)

    return topic


def get_element_random(filename):
    """
    Randomly selects a string element from a CSV file filename and returns it.
    :param filename: filename for a CSV file to randomly select an element from
    :return: random choice from the elements array from CSV filename
    """
    # Read in a CSV file and randomly selects an element from it
    with open(filename, 'r') as file:
        # Read the file content and split it by comma
        elements = file.read().split(',')

    # Strip white space
    elements = [element.strip() for element in elements if element.strip()]

    # Return random
    return random.choice(elements)


def get_llm_response(model, prompt, json=False, schema=None, config=None):
    octo_models = {
        "qwen1.5-32b-chat",
        "meta-llama-3-8b-instruct",
        "meta-llama-3-70b-instruct",
        "mixtral-8x22b-instruct",
        "nous-hermes-2-mixtral-8x7b-dpo",
        "mixtral-8x7b-instruct",
        "mixtral-8x22b-finetuned",
        "hermes-2-pro-mistral-7b",
        "mistral-7b-instruct",
        "llamaguard-7b",
        "codellama-7b-instruct",
        "codellama-13b-instruct",
        "codellama-34b-instruct",
        "llama-2-13b-chat",
        "llama-2-70b-chat"
    }

    azure_models = {
        'gpt-35-3',
        'gpt-40-1'
    }

    openai_models = {
        'gpt-4o',
        'gpt-4',
        'gpt-4-turbo',
        'gpt-3.5',
        'gpt-3.5-turbo'
    }

    groq_models = {
        'llama3-70b-8192',
        'llama3-8b-8192',
    }

    print(f"Using Model:", model)

    if model in openai_models:
        res = get_gpt_response(config, prompt, model)
    elif model in octo_models:
        try:
            res = get_octoml_response(config, prompt, model, schema=schema)
        except:
            res = get_octoml_response(config, prompt, model)
    elif model in groq_models:
        res = get_groq_response(config, prompt, model)
    elif model in azure_models:
        res = get_azure_response(config, prompt, model)
    else:
        raise ('unsupported model')

    # if json:
    # res = json_parser(config, res)

    return res


def get_azure_response(config, gpt_prompt, model, temperature=0.3, max_token=2000, response_type='text'):
    """
    Use Azure OpenAI's API to generate a response to the given prompt.

    Parameters:
        config (ConfigParser): The configuration parser object that contains the API key and endpoint.
        gpt_prompt (str): The prompt to pass to the API.
        model (str): The model to use.
        temperature (float): Controls the randomness of the output. A higher value makes the output more random, while a lower value makes it more deterministic.
        max_token (int): Controls the maximum length of the generated response.
        response_type (str): The response format type, either 'text' or 'json_object'.

    Returns:
        str: The content of the response generated by the API.
    """
    api_version = config.get('DEFAULT', 'AZURE_API_VER')

    # TODO: Fix this to be more elegant... The issue is that there are two different endpoints for different Azure served models...
    if model in ['gpt-35-3']:
        api_key = config.get('DEFAULT', 'AZURE_35_API_KEY')
        azure_endpoint = config.get('DEFAULT', 'AZURE_35_ENDPOINT')
    else:
        api_key = config.get('DEFAULT', 'AZURE_API_KEY')
        azure_endpoint = config.get('DEFAULT', 'AZURE_ENDPOINT')

    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=azure_endpoint
    )

    messages = [{"role": "system", "content": gpt_prompt}]

    args = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_token,
        "frequency_penalty": 0.0
    }

    if response_type == 'json_object':
        args["response_format"] = {"type": response_type}

    if 'instruct' in model:
        args["prompt"] = gpt_prompt
        response = client.completions.create(**args)
        content = response.choices[0].text  # The legacy way (as of Nov 8, 2023)
    else:
        args["messages"] = messages
        response = client.chat.completions.create(**args)
        content = response.choices[0].message.content

    return content


def generate_persona_seed():
    """
    Function that generates the SEED for the persona.
    NOTE: This only happens to UNINITIALIZED personas. This function will FAIL if the self.uuid is already set.
    :return: "seed" which is a JSON formatted SEED for persona creaetion
     """

    # Choose age randomly between 18 and 90
    # TODO: Make this selectable on demographic data
    age = random.randint(18, 90)

    # Select Random Gender
    # Define the list of gender identities
    gender_identities = ["Nonbinary", "Transgender", "Man",
                         "Woman"]  # https://www.pewresearch.org/short-reads/2022/06/07/about-5-of-young-adults-in-the-u-s-say-their-gender-is-different-from-their-sex-assigned-at-birth/
    # Define the corresponding gender weights
    # TODO: Make this selectable on demographic data
    gender_weights = [0.008, 0.008, 0.492, 0.492]
    # Select a gender identity based on the specified weights
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
                "National Affiliation": get_element_random('data/cultural_affiliations.txt'),
                "Positive Aspect": get_element_random('data/positive_adj.txt'),
                "Negative Aspect": get_element_random('data/negative_adj.txt'),
                "Education": education_level,
                "Wealth Index": wealth_level
            }
        }
    }

    # Convert to JSON
    seed = json.dumps(seed_dict, indent=4)

    # print(seed)

    return (seed)


def get_gpt_response(config, gpt_prompt, model='gpt-4o', response_type='text'):
    """
    Use OpenAI's API to generate a response to the given prompt.

    Parameters:
        config (ConfigParser): The configuration parser object that contains the API key.
        gpt_prompt (str): The prompt to pass to API
        model (str): The MODEL to use: SEE: https://platform.openai.com/docs/models

    Returns:
        str: The content of the response generated by API

    Note:
        - The temperature parameter controls the randomness of the output. A higher value makes the output more random, while a lower value makes it more deterministic.
        - The max_tokens parameter controls the maximum length of the generated response.
        - The frequency_penalty parameter can be used to reduce the likelihood of frequent words/phrases.
    """
    api_key = config.get('DEFAULT', 'OPENAI_API_KEY')
    client = OpenAI(api_key=api_key)

    messages = [{"role": "system",
                 "content": "You are a helpful assistant. You always do as you are instructed, and follow instructions precisely."},
                {"role": "user", "content": gpt_prompt}]

    # response_type_object = {"type": response_type}

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,  # description of temperature: http://bit.ly/3rmAJqu
        max_tokens=1000,
        frequency_penalty=0.0
        # response_format=response_type_object
    )

    # print(response)

    # print(gpt_prompt)

    # content = response['choices'][0]['message']['content'] #The legacy way (as of NOv 8, 2023)
    content = response.choices[0].message.content

    # Clean up JSON formatted string. Only needed sometimes...
    json_string = content.replace('```json', '')
    json_string = json_string.replace('```', '')
    json_string = json_string.strip()

    # Sometimes the following string(s) are included, which breaks the json mode. Fix if needed....

    # print()
    # print(content)

    # json_objects = json.loads(content)

    # return json_objects
    return json_string


def get_octoml_response(config, llm_prompt, model='mixtral-8x7b-instruct',
                        system_prompt="Below is an instruction that describes a task. Write a response that appropriately completes the request.",
                        schema=None):
    octoml_api_token = config.get('DEFAULT', 'OCTOML_API_KEY')

    client = Client(token=octoml_api_token)
    args = {
        "model": model,
        "max_tokens": 1000,
        "presence_penalty": 0,
        "temperature": 0.2,
        "top_p": 1,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": llm_prompt},
        ]
    }

    if schema:
        args['response_format'] = {"type": "json_object",
                                   "schema": schema.model_json_schema()}

    completion = client.chat.completions.create(**args)

    # Handle the response from the API
    message_content = completion.choices[0].message.content

    if schema:
        message_content = json.loads(message_content)
    return message_content


def get_groq_response(config, llm_prompt, model='llama3-8b-8192', system_prompt=""):
    api_key = config.get('DEFAULT', 'GROQ_API_KEY')
    client = Groq(api_key=api_key)

    chat_response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": llm_prompt},
        ]
    )
    return chat_response.choices[0].message.content


def get_hf_response(config, llm_prompt, model):
    parameters = {
        "max_new_tokens": 500
    }

    system_prompt = f"<SYS>You are a brilliant actor. You ALWAYS respond IN CHARACTER to the questions asked of you in a precise way. You do NOT need to re-state the question.</SYS> "
    prompt = (f"{system_prompt} "
              f"\n <PROMPT> \n"
              f"{llm_prompt}"
              f"</PROMPT>")

    # print(prompt)

    output = hf_query({
        "inputs": prompt,
        "parameters": parameters
    })

    print(output)

    generated_text = output[0]["generated_text"]

    # print(generated_text)

    return generated_text


def get_agent_row(uuid, cursor):
    # prevent SQL injection
    cursor.execute("SELECT * FROM agents WHERE uuid = %s LIMIT 1;", (uuid,))

    rows = cursor.fetchall()[0]

    #cursor.close()

    #print(rows)

    # Check if any rows were returned
    if rows:
        return rows
    else:
        return None

def get_memory_row(memory_uuid, cursor):
    # prevent SQL injection
    cursor.execute("SELECT * FROM memories WHERE uuid = %s LIMIT 1;", (memory_uuid,))

    rows = cursor.fetchall()[0]

    #cursor.close()

    #print(rows)

    # Check if any rows were returned
    if rows:
        return rows
    else:
        return None


def get_agent_list(cursor, do_print=False):
    cursor.execute("SELECT uuid, first_name, last_name FROM agents;")

    rows = cursor.fetchall()

    if do_print:
        for agent in rows:
            print(agent)

    # Check if any rows were returned
    if rows:
        return rows
    else:
        return None


def get_agent_summary(uuid, cursor):
    """
    Retrieves the summary of an agent from the database based on its UUID.

    This function fetches the summary for the agent with the given UUID from the
    'agents' table in the database. It is designed to support LLM prompting.
    If no UUID is provided or if the agent with the given UUID does not exist,
    the function returns None.

    :param uuid: The UUID of the agent to fetch the summary for.
    :type uuid: str or UUID
    :param cursor: The database cursor object used to execute SQL queries.
    :type cursor: psycopg2.extensions.cursor or similar

    :return: A tuple containing the summary for the agent, or None if not found.
    """

    if not uuid:
        print("No UUID provided. Fail")
        return None

    else:
        # retrieves an agent summary that can be used for quickly prompting the LLM
        query = sql.SQL("SELECT summary FROM agents WHERE uuid = %s LIMIT 1;")
        cursor.execute(query, [uuid])
        results = cursor.fetchall()
        if results:
            return results[0]
        else:
            return None


def get_stream(config, model='gpt-3.5-turbo'):
    api_key = config.get('DEFAULT', 'LLM_API_KEY')
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain asynchronous programming in the style of the pirate Blackbeard."},
        ],
        temperature=0,
        stream=True
    )

    for resp in response:
        sys.stdout.write(resp.choices[0].content)
        sys.stdout.flush()

    print(response.choices[0].message.content)

    print(json.dumps(json.loads(response.model_dump_json()), indent=4))

    print(response.choices[0].message.content)
