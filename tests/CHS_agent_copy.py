import base64
import io
import json
import random
import time
import uuid
from datetime import datetime
from decimal import Decimal
import gc
import ast

import PIL.Image
import faiss
import numpy as np
import pytz
import requests
import torch
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
from sentence_transformers import SentenceTransformer

from WorldEntity import WorldEntity
from utilities.logging_utils import send_log_message
from utilities.utils import get_activity_name, end_event, get_llm_response, generate_persona_seed, get_topic_random, \
    get_predicted_length, uuid_to_int64, get_last_event, store_tasks_to_db, get_task_details, get_tasks, \
    get_days_events, get_daily_activity_list, get_activity_random, find_activity, get_agent_row, gpu_check_avail_faiss, \
    store_objective_tasks_to_db, get_objective_intention, get_all_memory_vector_ids, are_lists_equal, \
    get_agent_event_activity_name, deactivate_agent_event_memories, reactivate_agent_event_memories, get_agent_event_row, \
    create_azure_client

from utilities.faiss_utils import create_faiss_index, get_faiss_manager, add_vectors, delete_faiss_index, search_index, \
    get_vector_ids

# Define the timezone (e.g., UTC)
#ToDo: Figure out time zone stuff (particularly for CompHuSim)
timezone = pytz.timezone('UTC')

#TODO: Trenton review this
#NOTE: This may cause deadlocks. In particular, this has to do with the forking of the process in ask_question (among others
#wherein the process of returning the response happens first, and then the memory is stored into the FAISS memory database
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Version 0.1 of the Comp-HuSim system.
# Development began in Fall 2023.
# Co-Creators: Dr. Trenton W. Ford (twford@wm.edu) and Dr. Michael Yankoski (myankosk@colby.edu)
# Collaborators:

class CompHuSimAgent:
    def __init__(self, config=None, logger = None, connection_pool=None, this_uuid='', do_create=True):
        #multiprocessing start method
        #multiprocessing.set_start_method('spawn')

        #print(f"THIS UUID:",this_uuid)

        # set up stuff: connections to LOGGING SERVER and FAISS DB SERVER
        self.config=config
        self.logging_host, self.logging_port = config.get('DEFAULT', 'LOGGING_HOST'), config.get('DEFAULT', 'LOGGING_PORT')
        self.faiss_host, self.faiss_port = config.get('DEFAULT', 'FAISS_HOST'), config.get('DEFAULT', 'FAISS_PORT')
        self.faiss_manager = get_faiss_manager(a=(self.faiss_host, int(self.faiss_port)), key=b'faiss')

        # Memory span for agent
        self.memory_span = 10  # how many memories to recall when "remembering" from FAISS DB (ie: k nearest neighbors)

        # Set up the current activity that the agent is engaged in.
        self.nullify_current_event()

        # Set up the current task list that the agent maintas.
        self.task_list = None

        # Model to use for agent's embedding memory
        # This is hot-swappable....
        # for more info see: https://www.sbert.net/docs/pretrained_models.html
        # self.embedding_model_name = 'all-MiniLM-L6-v2'
        self.embedding_model_name = 'all-mpnet-base-v2'
        self.model = SentenceTransformer(self.embedding_model_name)

        # Setup Connection Pool for Agent. Exit if fail.
        self.connection_pool = connection_pool
        if not self.connection_pool:
            print("Failed DB Connection. Exiting")
            send_log_message(f"ERROR: FAILED STARTUP: Error connecting to database. SHUTDOWN.",_host=self.logging_host, _port=self.logging_port)
            exit()

        # if a UUID is passed in, then we load this agent with those details from DB.
        # else spin up a new agent
        if this_uuid:
            # Check validity of UUID BEFORE we query DB
            #print(f"THIS UUID 2:", this_uuid)
            try:
                uuid.UUID(str(this_uuid))
                #print("IS VALID")
                self.uuid = this_uuid
                self.load_agent_details()
                #print(f"self.uuid",self.uuid)
            except ValueError as e:
                print(f"Failed to load agent: {e}")
                send_log_message(message=f"ERROR: FAILED TO LOAD AGENT: {e}",_host=self.logging_host, _port=self.logging_port)
                self.uuid = ''
        elif do_create:
            self.uuid = ''
            self.create()
            self.load_agent_details()
        else:
            self.create_agent_uuid()
            print(f"Created empty agent. UUID:", self.uuid)
            self.load_agent_details()

        self.client = create_azure_client(config=self.config, model='gpt-40-1')

    def get_traits(self):
        persona_traits = {
            "PERSON Summary": self.db_results['summary'],
            "PERSON Demographics": self.db_results['demographics'],
            "PERSON Personality": self.db_results['personality'],
            "PERSON Psychographics": self.db_results['psychographics'],
            "PERSON Family": self.db_results['family'],
            "PERSON Interests": self.db_results['interests']
        }
        return json.dumps(persona_traits, indent=4)

    def get_defined_subtask(self,activity_uuid=None,agent_event_id=None):
        #This function is specificially for Activities that have pre-defined subtasks.
        """

        Args:
            activity_uuid:
            agent_event_id:

        Kinds of Subtasks:
            OBSERVE
            MOVE
            INVENTORY_CHANGE
            SPEAK
            LOOKUP:     Call a function to utilize an external LLM

        Returns:

        """
        if activity_uuid is not None:
            lookup_activity_id = activity_uuid
        elif agent_event_id is not None:
            lookup_activity_id = agent_event_id
        else:
            print("Need some kind of UUID to lookup. Fail")
            return None

        print(f"Looking up",lookup_activity_id)



    def plan_activity_tasks(self, activity_uuid=None, task_uuid=None, store_to_db=True):
        # Get DB Stuff set up
        conn = self.connection_pool.getconn()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        if self.current_event:
            activity_name = self.current_event['activity_name']
            activity_uuid = self.current_event['activity_uuid']
            agent_event_uuid = self.current_event['agent_event_uuid']
        elif activity_uuid:
            activity_name = get_activity_name(cursor, activity_uuid)
        elif task_uuid:
            task_details = get_task_details(cursor, task_uuid)
        else:
            print("No activity currently engaged AND no parent task engaged. Therefore no tasks to plan.")
            return None

        completed_task_list = ""

        llm_prompt = f"""
        Imagine you are the following person: {self.get_traits()}

        You are currently engaged in the following activity: {activity_name}.

        The current tasks you have engaged in thus far related to this activity are: {completed_task_list}.

        Your job is to generate a JSON FORMATTED TASK LIST of SUCCINCT STEPS needed to complete the activity.

        Format your TASK LIST like this: {{
            "1": {{"step": "<step1 succinct summary>", "reason": "<justification for step1>"}},
            "2": {{"step": "<step2 succinct summary>", "reason": "<justification for step2>"}},
            "3": {{"step": "<step3 succinct summary>", "reason": "<justification for step3>"}},
            ...
        }}
        """

        #print(llm_prompt)

        plan = get_llm_response(model='gpt-4o', prompt=llm_prompt, config=self.config, client=self.client)

        #Store to DB if asked to do so
        if store_to_db:
            store_tasks_to_db(cursor=cursor,
                              agent_uuid=self.uuid,
                              agent_event_uuid=agent_event_uuid,
                              task_json_string=plan)

        #Close DB Stuff
        cursor.close()
        self.connection_pool.putconn(conn)

        return plan

    def get_days_event_history(self):
        # Get DB Stuff set up
        conn = self.connection_pool.getconn()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        activities = get_days_events(cursor=cursor, agent_uuid=self.uuid)

        cursor.close()
        self.connection_pool.putconn(conn)

        return activities

    def get_last_event(self,do_load_unfinished_tasks=False):
        # Get DB Stuff set up
        conn = self.connection_pool.getconn()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        last_event = get_last_event(cursor=cursor, agent_uuid=self.uuid)

        if do_load_unfinished_tasks:
            self.load_unfinished_tasks()

        cursor.close()
        self.connection_pool.putconn(conn)

        return last_event

    def load_unfinished_tasks(self):
        last_event = self.get_last_event()
        if last_event['ended_at'] is None:
            last_event_uuid = last_event['uuid']
        else:
            print("I have no unfinished activities.")

        conn = self.connection_pool.getconn()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        print(f"Loading tasks for last event; uuid: {last_event_uuid}")

        self.task_list = get_tasks(cursor=cursor, agent_event_uuid=last_event['uuid'])

        cursor.close()
        self.connection_pool.putconn(conn)


    def choose_next_activity(self,activities_list=None):
        #WARNING: The list of possible activities needs to be a SUBSET from daily_activities table. Otherwise there will be downstream failures
        #   (ie: in getting the activity_name from the daily_activities table)

        # Get DB Stuff set up
        conn = self.connection_pool.getconn()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        #ToDo: Expand this to include the intention in this function call, if it considers memories here.

        if activities_list is None:        #If no possible activities are passed in, then we get possible activities from the daily_activitys list.
            # Get Daily Activities List
            daily_activities_list = get_daily_activity_list(cursor)
            # Prepare the list of activities as a string for the prompt
            activities_list = '\n'.join([f"UUID: {row['uuid']}, Activity Name: {row['activity_name']}" for row in daily_activities_list])

        #Get current activity name / description etc:
        if self.current_event is None:
            current_activity_name = "NO ACTIVITY"
            current_activity_intention = "NO ACTIVITY"
            start_time = ""
        else:
            current_activity_name = self.current_event['activity_name']
            start_time = self.current_event['start_time']
            if self.current_event.get('intention') is None:
                current_activity_intention = "NO ACTIVITY"
            else:
                current_activity_intention = "For this reason: \"" + self.current_event['intention'] + "\""

        #Get last activity information, based on the last agents event
        last_event = get_last_event(cursor=cursor, agent_uuid=self.uuid, offset = 1)
        if last_event:
            last_activity_name = get_activity_name(cursor=cursor,activity_uuid=last_event['activity_uuid'])
            last_activity_intention = "For this reason: \""+ last_event['intention'] + "\""
        else:
            last_activity_name = ""
            last_activity_intention = ""

        current_time = datetime.utcnow()

        llm_prompt = f"""
                        Imagine that YOU are the PERSON described by the TRAITS below. You are deciding what to do next.
                        Currently, you are engaged in this CURRENT ACTIVITY: "{current_activity_name.upper()}". "{current_activity_intention}".
                        You started this activity at {start_time}. It is currently {current_time}.

                        Prior to that, you were engaged in these PREVIOUS ACTIVITIES: "{get_days_events(cursor, self.uuid)}".

                        Your choice of what to do next should be aligned with your TRAITS: personality, demographics, psychographics, etc.

                        Your TRAITS are as follows:

                        {self.get_traits()}

                        Based on your CURRENT ACTIVITY, and your PREVIOUS ACTIVITIES, consider the list of POSSIBLE ACTIVITIES.

                        Respond with ONLY the UUID of the NEXT ACTION that you choose to do, which is in alignment with your TRAITS.

                        Do NOT repeat the same activity that you are currently engaged in.

                        Here is the list of POSSIBLE ACTIVITIES:
                        \n{activities_list}
                        \n
                        Return ONLY the text of the UUID of the ACTIVITY that you are choosing to engage next. Like this: 00000000-0000-0000-0000-000000000000
        """

        #print(llm_prompt)

        response = get_llm_response(model='gpt-4o', prompt=llm_prompt, config=self.config, client=self.client)

        cursor.close()
        self.connection_pool.putconn(conn)

        return response

    def start_event(self, pick_random_daily_activity=False, activity_uuid=None, activity_name=None, do_find_activity = False, activity_description=None, create_daily_activity=False, generate_intention=False, intention=None, context_info=None):
        """
        Start a new activity and record it in the database.

        Parameters:
        activity_uuid (str, optional): The unique identifier for the activity. Defaults to None.
        activity_description (str, optional): A description of the activity. Defaults to None.
                                          If set, the system tries to find an activity UUID that matches.
                                          If create_activity is True, if the system doesn't find a matching UUID for activity_description
                                          it will attempt to create a new daily activity for this activity_description.
        create_activity (bool, optional): If True, will add the new activity to the daily_activities table
        generate_intention (bool, optional): If True, generates and sets an intention for the activity. Defaults to False.
        pick_random (bool, optional): If True, starts a random activity for the agent. Defaults to False.

        Returns:
        dict: A dictionary containing the details of the current activity:
          - 'agent_event_uuid' (str): The UUID of the new agent event.
          - 'activity_uuid' (str): The UUID of the activity.
          - 'activity_name' (str): The name of the activity.
          - 'intention' (str, optional): The intention for the activity, if defined.
        """

        # Get DB Stuff set up
        conn = self.connection_pool.getconn()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        #End current event if one already exists for agent. IE: we're moving onto the next thing.
        self.end_current_event()

        if activity_name is None and activity_uuid is None and activity_description is None and pick_random_daily_activity is False:
            send_log_message(f"ERROR: Can't start activity without activity_name, activity_uuid, activity_description OR pick_random",_host=self.logging_host, _port=self.logging_port)
            return None

        #Figure out or generate an activity_UUID based on pick_random or activity_description
        if pick_random_daily_activity:
            activity_uuid = get_activity_random(cursor)['uuid']
            activity_name = get_activity_name(cursor, activity_uuid=activity_uuid)
        elif activity_description is not None and do_find_activity:
            activity_uuid = find_activity(cursor=cursor,activity_description=activity_description, config=self.config, create_activity=create_daily_activity)
            activity_name = get_activity_name(cursor, activity_uuid=activity_uuid)

        else:
            send_log_message(f"ERROR: Didn't have enough information to start Agent Event.")

        #Get the activity name:
        if activity_name is None:
            send_log_message(f"ERROR: No Event Activity Name Specified. FAIL.")
            return None

        # Create the insert query to add the Activity to the DB
        insert_query = """
                        INSERT INTO agent_events (belongs_to,activity_uuid,activity_name,context_info)
                        VALUES (%s, %s, %s, %s)
                        RETURNING uuid
        """

        #Execute DB Insert
        values_to_insert = (self.uuid, activity_uuid, activity_name, context_info)
        cursor.execute(insert_query, values_to_insert)
        new_event_uuid = cursor.fetchone()['uuid']
        conn.commit()

        #If requested to generate_intention, generate an AGENT_SPECIFIC intention for the activity, and update it in the DB.
        if generate_intention:
            #Do the work to generate the agent-specific intention if requested in function call
            intention = self.generate_event_intention(agent_event_uuid=new_event_uuid)

        if intention is not None:
            update_query = """
                        UPDATE agent_events
                        SET intention = %s
                        WHERE uuid = %s
            """
            values_to_update = (intention,new_event_uuid)
            cursor.execute(update_query, values_to_update)
            conn.commit()
            #print(intention)
        else:
            intention = None

        #Set the current timestamp
        current_timestamp_with_tz = datetime.now(timezone)
        formatted_timestamp = current_timestamp_with_tz.strftime('%Y-%m-%d %H:%M:%S.%f %z')

        #define and update the dictionary for self.current_activity
        current_activity = {'agent_event_uuid':new_event_uuid,
                            'activity_uuid':activity_uuid,
                            'activity_name':activity_name,
                            'context_info':context_info,
                            'start_time':formatted_timestamp}

        #if we have generated an intention for the current activity, add that to the dictionary as well
        if intention:
            current_activity['intention']=intention

        if context_info:
            current_activity['context_info']=context_info

        self.current_event = current_activity

        #Cleanup and return new_event_uuid
        conn.commit()
        cursor.close()
        self.connection_pool.putconn(conn)

        return current_activity

    def initialize_objective(self, intention=''):
        #THis function creates a new agnet_event (agent_events table), but the type is an <OBJECTIVE> IE: A project to be completed.
        #This is separate from an ACTIVITY  (ie: something contained in the Daily Activities table).
        #The idea here is that these can be created arbitrarily, whereas the DailyActivities are constrained to Activities.

        # Get DB Stuff set up
        conn = self.connection_pool.getconn()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Create the insert query to add the Activity to the DB
        insert_query = """
                               INSERT INTO agent_events (belongs_to,intention)
                               VALUES (%s, %s)
                               RETURNING uuid
               """

        # Execute DB Insert
        values_to_insert = (self.uuid, intention)
        cursor.execute(insert_query, values_to_insert)
        new_objective_uuid = cursor.fetchone()['uuid']
        conn.commit()

        # Set the current timestamp
        current_timestamp_with_tz = datetime.now(timezone)
        formatted_timestamp = current_timestamp_with_tz.strftime('%Y-%m-%d %H:%M:%S.%f %z')

        # define and update the dictionary for self.current_activity
        current_activity = {'agent_event_uuid': new_objective_uuid, 'activity_name': intention, 'start_time': formatted_timestamp}

        self.current_event = current_activity

        return new_objective_uuid

    def plan_objective_tasks(self, objective_uuid=None, environment='', task_uuid=None, store_to_db=True):
        # Get DB Stuff set up
        conn = self.connection_pool.getconn()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        objective_intention = get_objective_intention(cursor=cursor,objective_uuid=objective_uuid)
        print(f"Objective Intention:",objective_intention)

        #ToDo. Should return memories associated with completed tasks
        completed_task_list = ""

        llm_prompt = f"""
        Imagine you are the following person: {self.get_traits()}

        You are currently active with the following OBJECTIVE: {objective_intention}.

        The current tasks you have engaged in thus far related to this OBJECTIVE are: {completed_task_list}.

        The ENVIRONMENT you find yourself in is as follows: {environment}.

        Your job is to generate a JSON FORMATTED TASK LIST of SUCCINCT STEPS needed to complete the activity.

        Keep in mind that you may OBSERVE or EXAMINE any ENTITY in your ENVIRONMENT--including the ENVIRONMENT itself--in order to accomplish your OBJECTIVE.

        If you are SEARCHING for something, it will be best to EXAMINE everything in your ENVIRONMENT to be sure you aren't missing anything.

        When you OBSERVE an ENTITY you will receive only basic information about it. This means you may not notice everything.

        However, when you EXAMINE an ENTITY, you will get more detailed information about it.

        To refer to a specific ENTITY in your ENVIRONMENT, refer to it by its UUID.

        For each ENTITY, choose one of two ACTIONS: OBSERVE or EXAMINE.

        EXAMINING each entity will provide additional information to make decisions with.

        Format your TASK LIST like this: {{
            "1": {{"command": "<ACTION>", "uuid":"<entity_uuid>","reason": "<justification for step1>"}},
            "2": {{"command": "<ACTION>", "uuid":"<entity_uuid>","reason": "<justification for step2>"}},
            "3": {{"command": "<ACTION>", "uuid":"<entity_uuid>","reason": "<justification for step3>"}},
            ...
        }}

        However, if you have COMPLETED your OBJECTIVE, SIMPLY respond with a JSON object like this:
        {{"STATUS":"COMPLETE","JUSTIFICATION":"<JUSTIFICATION FOR WHY YOU THINK TASK IS COMPLETE>"}}
        """

        #print(llm_prompt)

        plan = get_llm_response(model='gpt-4o', prompt=llm_prompt, config=self.config, client=self.client)

        print(plan)

        #Store to DB if asked to do so
        if store_to_db:
            store_objective_tasks_to_db(cursor=cursor,
                              agent_uuid=self.uuid,
                              agent_event_uuid=objective_uuid,
                              task_json_string=plan)

        #Close DB Stuff
        cursor.close()
        self.connection_pool.putconn(conn)

        return plan

    def do_objective(self,objective_uuid,environment=''):
        # Get DB Stuff set up
        conn = self.connection_pool.getconn()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        #Do Query Stuff
        sql="""select * from agent_tasks where agent_event_uuid = %s"""
        cursor.execute(sql,(objective_uuid,))
        results = cursor.fetchall()

        objective_intention = get_objective_intention(cursor, objective_uuid=objective_uuid)

        #Loop through the results and do the relevant actions
        for result in results:
            print(f"Attempting to do this:",result['task_description'],"for this reason: ",result['rationale'])

            task_elements = json.loads(result['task_description'])
            task_entity_uuid = task_elements['uuid']
            task_entity_command = str(task_elements['command']).upper()

            #Now do some fancy stuff: ie: get the World Entity & do stuff with it! WOO HOO!!! POWER TO THE AI!!!
            entity = WorldEntity(connection_pool = self.connection_pool,this_uuid = task_entity_uuid)

            #ToDo: Other kinds of ACTIONS that can be done in the environment
            if task_entity_command == 'OBSERVE':
                command_result = entity.be_observed()
            elif task_entity_command == 'EXAMINE':
                command_result = entity.be_examined()
                #Recursive call into planning process... Might get expensive... But, how else to do this?
                #subsequent_objective_uuid = self.initialize_objective(intention = objective_intention + ' subtask')
                #self.plan_objective_tasks(objective_uuid=subsequent_objective_uuid,environment=command_result)
                #self.do_objective(objective_uuid=subsequent_objective_uuid,environment=command_result)

            #Now evaluate whether the task has been completed based on the Environment
            llm_prompt = f"""
            You are attempting to complete the following OBJECTIVE: {objective_intention}
            Within this ENVIRONMENT: {environment}
            You have just completed the following TASK: {result['task_description']}
            The RESULT of this task is: {command_result}

            If there are MORE ENTITIES that you want to examine contained in the RESULT, return a JSON OBJECT of those tasks like this:
            {{
                "STATUS": "NEWTASK",
                "TASKS": {{
                    "1": {{"command": "<ACTION>", "uuid":"<entity_uuid>","reason": "<justification for step1>"}},
                    "2": {{"command": "<ACTION>", "uuid":"<entity_uuid>","reason": "<justification for step2>"}},
                    "3": {{"command": "<ACTION>", "uuid":"<entity_uuid>","reason": "<justification for step3>"}},
                    ...
                }}
            }}

            If you have completed your OBJECTIVE, return a JSON object like this:
            {{
                "STATUS": "COMPLETE",
                "JUSTIFICATION": "<JUSTIFICATION FOR WHY YOU THINK TASK IS COMPLETE>"
            }}

            If you have NOT completed your OBJECTIVE, return an empty JSON object.
            """

            llm_response_json = json.loads(get_llm_response(model='gpt-4o',prompt=llm_prompt, config=self.config, client=self.client))

            #print("Evaluating the result of the TASK...")

            if llm_response_json.get('STATUS') == "COMPLETE":
                #print(f"RESULT:", command_result)
                print()
                print(f"SUCCESS!!!! The task completed successfully.",llm_response_json.get('JUSTIFICATION'))
                print()
                cursor.close()
                self.connection_pool.putconn(conn)
                return llm_response_json

            elif llm_response_json.get('STATUS')=="NEWTASK":
                print()
                print("ADDING NEW TASKS!!!:")
                new_tasks = llm_response_json.get('TASKS')
                print(f"NEW TASKS:",new_tasks)
                print()

                #ToDo: This needs to add a new OBJECTIVE & OBJECTIVE_TASKS in order to pursue additional infrmation.
                #Right now, the agent's FAIL to accomplish tasks that are "behind" OBSCURED World Entities.


            del entity

        print()
        print(f"FAIL: The Objective was NOT completed:",objective_intention)
        print()

        cursor.close()
        self.connection_pool.putconn(conn)

    def end_current_event(self):
        """
        Ends the agent's current activity and updates the database accordingly.

        This function performs the following steps:
        1. Sets up the database connection and cursor using the connection pool.
        2. Summarizes the event by drawing upon the associated memories (to be implemented).
        3. Checks if there is a current activity set for the agent.
        4. If a current activity exists, it calls the `end_activity` function to set the 'ended_at' timestamp for the activity and commits the transaction.
        5. Closes the cursor and returns the connection to the connection pool.
        6. Resets the agent's `current_activity` to `None`.

        Args:
            None

        Returns:
            None
        """
        # Get DB Stuff set up
        conn = self.connection_pool.getconn()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        #ToDo: Here this code should SUMMARIZE THE EVENT by drawing upon the memories that are associated with it

        #Only do this if there is a current activity set
        if self.current_event is not None:
            end_event(cursor, self.current_event['agent_event_uuid'])
            conn.commit()

        cursor.close()
        self.connection_pool.putconn(conn)

        #Finally, set the agent's current_activity to nothing.
        self.nullify_current_event()

    def nullify_current_event(self):
        self.current_event = {}
        self.current_event['activity_name'] = "NO ACTIVITY"
        self.current_event['activity_uuid'] = None
        self.current_event['intention'] = ""
        self.current_event['agent_event_uuid'] = None
        self.current_event['context_info'] = ""

    def set_current_event(self,agent_event_uuid=None):
        # Get the event from the DB
        if agent_event_uuid is None:
            print(f"Error: No Agent Event UUID provided.")
            send_log_message(message="ERROR: NO Agent Event UUID Provided.")

        # Get DB Stuff set up
        conn = self.connection_pool.getconn()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        query = """
                    SELECT  ae.uuid,
                            ae.activity_name,
                            ae.activity_uuid,
                            ae.intention,
                            ae.context_info,
                            TO_CHAR(ae.started_at AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS') as started_at,
                            TO_CHAR(ae.ended_at AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS') as ended_at
                    FROM agent_events ae
                    WHERE ae.uuid = %s and belongs_to = %s
                """

        cursor.execute(query, (agent_event_uuid,self.uuid))

        result = cursor.fetchone()

        # Setup agent's current event (only if a result is returned)
        if result:
            self.current_event = {
                'uuid': result['uuid'],
                'activity_name': result['activity_name'],
                'activity_uuid': result['activity_uuid'],
                'agent_event_uuid': agent_event_uuid,
                'intention': result['intention'],
                'context_info': result['context_info'],
                'started_at': result['started_at'],
                'ended_at': result['ended_at']
            }
        else:
            self.nullify_current_event()
            print(f"Error: No Agent Event found from provided UUID.")
            send_log_message(message="ERROR: No Agent Event found from provided UUID.")

        cursor.close()
        self.connection_pool.putconn(conn)

    def generate_event_intention(self, activity_uuid=None, agent_event_uuid=None, model='gpt-4o-mini-2024-07-18'):
        """
            Defines an intention for an activity based on the agent's persona using a language model.

            This function performs the following steps:
            1. Establishes a connection to the database and creates a cursor.
            2. Retrieves the activity name based on either the provided `activity_uuid` or `agent_event_uuid`.
            3. Constructs a prompt to define the agent's intention for engaging in the activity, incorporating the agent's persona details.
            4. Sends the prompt to a language model to generate a response that articulates the agent's intention.
            5. Returns the generated intention.

            Parameters:
            activity_uuid (str, optional): The unique identifier for the activity. Defaults to None.
            agent_event_uuid (str, optional): The unique identifier for the agent event. Defaults to None.

            Returns:
            str: The intention generated by the language model that is AGENT SPECIFIC.

            Note:
            - This function assumes that `self.db_results` contains the persona details including 'summary', 'demographics', 'family', 'psychographics', and 'personality'.
        """

        # Get DB Stuff set up
        conn = self.connection_pool.getconn()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # LLM Prompt to Define a PERSONA based INTENTION for Activity
        if activity_uuid:
            activity_name = get_activity_name(cursor, activity_uuid=activity_uuid)
        elif agent_event_uuid:
            activity_name = get_agent_event_activity_name(cursor, agent_event_uuid=agent_event_uuid)

        llm_prompt = f"""
                        Imagine that YOU are the PERSON described by the TRAITS below. You are just about to begin this activity: {activity_name}.
                        Respond with a SHORT articulation of your INTENTION for engaging in this activity. Why are you choosing to do this?
                        Your answer should take into account relevant aspects and TRAITS of who you are, your psychological nature, your personality, and your interests.

                        Your TRAITS are as follows:

                        {self.get_traits()}
        """

        response = get_llm_response(model=model,prompt=llm_prompt,config=self.config, client=self.client)

        cursor.close()
        self.connection_pool.putconn(conn)

        return response

    def prune_outlier_memories(self, ids, distances, max_results=5):
        '''Returns a pruned set (max set size = max_results) of memories based on IRQ distribution calculations
        '''
        # Calculating IQR for distance pruning
        # ToDo: Experiment with and determine the best way to set these values
        Q1 = np.percentile(distances, 40)
        Q3 = np.percentile(distances, 60)
        IQR = Q3 - Q1

        # Defining bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        print(f"Bounds:", lower_bound, " ", upper_bound)

        # Filtering out outliers
        pruned_ids = []
        pruned_distances = []
        for id, distance in zip(ids, distances):
            if lower_bound <= distance <= upper_bound:
                pruned_ids.append(id)
                pruned_distances.append(distance)
                # Stop adding more IDs if the limit is reached
                if len(pruned_ids) >= max_results:
                    break

        return pruned_ids, pruned_distances

    def faiss_remember(self, search_string, nearest_neighbors=None, print_time=False, llm_check_memories=False, debug=False):
        """
        A function for recalling the most relevant memories from the agent's memory list.
        :param search_string:
        :return: a dictionary of memory objects from the memories table
        """
        # TODO: uncomment the logger info and helper info
        # send_log_message(f"INFO: LOADING AGENT MEMORY {self.uuid}",_host=self.logging_host, _port=self.logging_port)
        # print("LOADING AGENT MEMORY " + self.uuid)

        # Retrieve relevant memories with Timer
        start_time = time.time()

        # If we don't set the nearest_neighbors that we want when calling this function, then default to the agent's self.memory_span as default
        if nearest_neighbors is None:
            nearest_neighbors = self.memory_span
            send_log_message(f"Using default K Nearest Neighbors Setting: {nearest_neighbors}")

        # Create an embedding for the search_string
        new_embedding = self.model.encode([search_string], convert_to_tensor=True)
        new_embedding = new_embedding.cpu().detach().numpy()  # Convert to numpy array, perform on GPU

        # Searching in the index for the top nearest neighbors
        #D, I = self.faiss_memories.search(new_embedding.astype('float32'), nearest_neighbors)
        send_log_message(f"Trying to search memory index for: \"{search_string}\", with K nearest neighbors: {nearest_neighbors}")
        D, I = search_index(self.faiss_manager, self.uuid, new_embedding.astype('float32'), nearest_neighbors)

        # Retrieve the IDs and Distances associated with the nearest neighbors
        nearest_neighbor_ids = I[0].tolist()
        nearest_neighbor_distances = D[0].tolist()
        original_num_memories = len(nearest_neighbor_ids)

        if debug:
            print(f"Found", original_num_memories, "potentially relevant Agent memories based on FAISS similarity search.")
            # Get the relevant memories from the Postgres DB
            original_relevant_memories = self.get_memories_dict(memory_vector_ids=nearest_neighbor_ids)
            print(f"Relevant Memories (Original): {original_relevant_memories}\n")

        # Prune the outlier memories based on distance scores
        pruned_ids, pruned_distances = self.prune_outlier_memories(nearest_neighbor_ids, nearest_neighbor_distances)
        if debug: print(f"Pruning down to ", len(pruned_ids), "relevant memories.")

        # Display the results
        # for id, distance in zip(nearest_neighbor_ids, nearest_neighbor_distances):
        #    print(f"ID: {id}, Distance: {distance}")

        # Get the relevant memories from the Postgres DB
        pruned_relevant_memories = self.get_memories_dict(memory_vector_ids=pruned_ids)

        # Create a mapping of IDs to distances
        id_distance_map = {Decimal(id): distance for id, distance in
                           zip(pruned_ids, pruned_distances)}

        # Iterate through the relevant_memories and add the corresponding distance
        for memory_id in pruned_relevant_memories:
            # Check if this memory's ID is in the id_distance_map
            if memory_id in id_distance_map:
                pruned_relevant_memories[memory_id]['distance'] = id_distance_map[memory_id]

        relevant_memories = pruned_relevant_memories

        if debug: print(f"Relevant Memories (Pruned): {relevant_memories}\n")

        end_time = time.time()


        # Calculate and display elapsed time
        elapsed_time = end_time - start_time
        if print_time or debug:
            print(f"Elapsed time: {elapsed_time} seconds")

        send_log_message(f"INFO: FAISS-based CompHuSimAgent.remember() Elapsed Time: {elapsed_time:.5g} sec",_host=self.logging_host, _port=self.logging_port)

        #Use LLM to confirm relevance of returned memories
        if llm_check_memories and len(original_relevant_memories)>1:
            #This adds another layer of LLM-powered selection of potentially relevant memories.
            # Note that this isn't in addition to the FAISS search, but rather LIMITS the results from the ORIGINAL FAISS DB search (ie: the largest original FAISS search).

            prompt = (f"Based on the following POSSIBILITIES, return A LIST (ie: ['uuid1','uuid2','uuid3'] of ALL UUIDs for any of the objects related to: \"{search_string}\". "
                      f"Here is the list of POSSIBILITIES: \"{original_relevant_memories}\"")
            #print(prompt)

            #Get LLM response and turn it into an actual list.
            llm_identified_memory_uuids = ast.literal_eval(get_llm_response(model='gpt-4o',prompt=prompt,config=self.config, client=self.client))
            print(llm_identified_memory_uuids)
            if debug: print(f"LLM Identified Memory UUIDS:{llm_identified_memory_uuids}")

            # Filter relevant_memories based on the UUIDs returned by the LLM as relevant
            relevant_memories = {k: v for k, v in original_relevant_memories.items() if v['uuid'] in llm_identified_memory_uuids}

            # Calculate and display elapsed time
            end_time = time.time()
            elapsed_time = end_time - start_time
            if print_time or debug:
                print(f"LLM identified {len(list(llm_identified_memory_uuids))} relevant memories from ORIGINAL list. Elapsed Time: {elapsed_time:.5g} sec")

            #Log Elapsed Time
            send_log_message(f"LLM Check Memories: Elapsed Time: {elapsed_time:.5g} sec")

        if debug: print(f"Final List of {len(relevant_memories)} relevant_memories: {relevant_memories}")

        return relevant_memories

    def get_memories_dict(self, memory_vector_ids=None, event_context_uuid=None, debug=False):
        """
            Retrieves a dictionary of memory records from the 'memories' table in the PostgreSQL database
            where the 'belongs_to' column matches the agent's UUID. If 'memory_ids' is provided, only the
            memories with 'vector_db_id' contained in 'memory_ids' are retrieved.

            Parameters:
            - memory_ids (list of int, optional): A list of 'vector_db_id's to filter the memories returned.
                                                  If not provided, all memories belonging to the agent's UUID
                                                  will be retrieved.

            Returns:
            - dict: A dictionary where each key is the 'vector_db_id' of a memory and each value is another
                    dictionary containing details of the memory (such as 'uuid', 'description', 'summary',
                    'keywords', 'intensity', 'primary_emotions').

            The function initializes a connection to the database and creates a cursor with a dictionary-like
            interface. It constructs an SQL query that either selects all records associated with the agent's UUID
            or filters these records based on the provided 'memory_ids'. After executing the query, it fetches all
            results and populates a dictionary with the data. The connection is returned to the pool before the
            function returns the dictionary of memories.
        """

        # DB Stuff
        conn = self.connection_pool.getconn()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Base SQL statement to get memories from Postgres Table
        query = """
                    SELECT vector_db_id, uuid, description, summary, keywords, intensity, primary_emotions, created_at
                    FROM memories
                    WHERE belongs_to = %s and deactivated = %s
                    """

        # Initialize a list for the parameters that will be passed to the cursor.execute() method
        params = [self.uuid,0]

        # If memory_ids is provided, modify the query to filter by those IDs
        if memory_vector_ids is not None:
            # Append an AND clause to match the vector_db_id with any of the ids in memory_ids
            # TODO: Is this going to be a sufficiently efficient way to query the DB when the memories table is large?
            query += "AND vector_db_id = ANY(%s) ORDER BY created_at DESC;"
            params.append(memory_vector_ids)
        elif event_context_uuid is not None:
            # Append an AND clause to match the given memory with the event_context_uuid
            #ToDo: Make this capable of handling a list of agent_event_uuids
            query += "AND agent_event_uuid = %s ORDER BY created_at DESC;"
            params.append(event_context_uuid)
        else:
            # If memory_ids is not provided, end the current query
            query += "ORDER BY created_at DESC;"

        if debug: print(f"GET MEMORIES DICT QUERY: {query}")

        memories_dict = {}

        # Execute the query with agent's UUID
        cursor.execute(query, params)

        # Create dict with results
        rows = cursor.fetchall()
        send_log_message(f"Loading", len(rows), "memories for agent.")
        for row in rows:
            # Using vector_db_id as the dictionary key
            memories_dict[row['vector_db_id']] = {
                'uuid': row['uuid'],
                'summary': row['summary'],
                'description': row['description'],
                'created_at': row['created_at'],
                'keywords': row['keywords'],
                'intensity': row['intensity'],
                'primary_emotions': row['primary_emotions']
            }

        # Clean Up & return DB connection to pool
        cursor.close()
        self.connection_pool.putconn(conn)

        return memories_dict

    def sync_memory_vector_db(self,dimension=None,debug=False,recreate=False):
        """
        This function checks to ensure that the PostgreSQL db vector_db_id list is equal to the faiss db vector_db_id list for this agent.
        If not, then it reloads the missing memories into the vector_db.
        However: if recreate == True, then we DELETE the faiss db index and recreate from scratc
        """

        #Set up embedding dimensions:
        if dimension is None:
            dimension = self.model.get_sentence_embedding_dimension()

        # Get DB Stuff set up
        conn = self.connection_pool.getconn()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        #get vector_db lists for comparison
        faiss_vector_db_id_list = get_vector_ids(self.faiss_manager,self.uuid)
        postgres_vector_db_id_list = get_all_memory_vector_ids(self.uuid,cursor)

        if debug:
            print(f"Postgres Vector DB ID List: {len(list(postgres_vector_db_id_list))}\n{list(postgres_vector_db_id_list)}\n")
            print()
            print(f"Faiss Vector DB ID List: {len(list(faiss_vector_db_id_list))}\n{list(faiss_vector_db_id_list)}\n")

        #Recreate if desired...
        if recreate:
            send_log_message(f"Attempting to recreate VectorDB Index from Postgres. Deleting & Creating afresh...")
            delete_faiss_index(self.faiss_manager, self.uuid)
            create_faiss_index(self.faiss_manager, self.uuid, dimension)
            faiss_vector_db_id_list = {}

        if postgres_vector_db_id_list is None:
            if debug: print(f"INFO: Postgres DB is EMPTY. Stop execution here.")
            send_log_message(f"INFO: Postgres DB is EMPTY. Stop execution here.")
            return

        # check if lists are equivalent
        if are_lists_equal(postgres_vector_db_id_list,faiss_vector_db_id_list):
            if debug: print(f"INFO: Vector DB Lists are EQUAL")
            send_log_message(f"INFO: Vector DB Lists are EQUAL")
        else:
            memories_to_add_to_faiss = are_lists_equal(postgres_vector_db_id_list,faiss_vector_db_id_list,type='diff')
            send_log_message(f"INFO: Vector DB Lists are NOT EQUAL. Need to add {len(memories_to_add_to_faiss)} memories to FAISS index.")

            if debug:
                print(f"INFO: Vector DB Lists are NOT EQUAL. Need to add {len(memories_to_add_to_faiss)} memories to FAISS index.")
                print(f"Need to add these memories to FAISS DB Index:{memories_to_add_to_faiss}")
                for vector_db_id in memories_to_add_to_faiss:
                    print(f"Need to add this to Faiss Vector DB ID:",vector_db_id)

            #Get the memories that need to be added to the Faiss DB Server
            memories_dict = self.get_memories_dict(memory_vector_ids=list(memories_to_add_to_faiss))

            if debug: print(f"ADDING THESE {len(memories_dict)} Memories Dict to VECTOR DB:", memories_dict, "\n\n")

            # Create a new dictionary with just the IDs and DESCRIPTIONS from the memories
            mem_descriptions_dict = {
                str(id): ' '.join([
                    memory.get('description', '').strip(),
                    memory.get('keywords', '').strip(),
                    memory.get('primary_emotions', '').strip()
                ]).strip()
                for id, memory in memories_dict.items()
            }

            if debug: print("MEM Descriptions DICT:", mem_descriptions_dict, "\n\n")

            # Create a new list of just the memory descriptions (without ID keys)
            descriptions_list = list(mem_descriptions_dict.values())

            if debug: print("Descriptions LIST:", descriptions_list, "\n\n")

            # Convert descriptions to embeddings
            embeddings = self.model.encode(descriptions_list, convert_to_tensor=True)

            # Initialize FAISS index
            dimension = embeddings.shape[1]
            if debug: print(f"Embedding dimension2: {dimension}")

            # Convert to numpy array
            embeddings = embeddings.cpu().detach().numpy()
            if debug: print("Embeddings shape:", embeddings.shape)
            if debug: print("Embeddings dtype:", embeddings.dtype)
            if debug: print("Made it to here 2")

            # Extract the IDs from the memory descriptions dictionary
            ids = np.array([id for id in mem_descriptions_dict.keys()], dtype=np.int64)
            if debug:
                print("Made it to here 4")
                print("IDs shape:", ids.shape)
                print("IDs dtype:", ids.dtype)
                print("IDs:", ids)

            # Ensure embeddings and ids have matching lengths
            if embeddings.shape[0] != ids.shape[0]:
                raise ValueError("The number of embeddings does not match the number of IDs.")

            if debug: print("Embeddings:", embeddings)

            # Add the embeddings to the FAISS index with their corresponding IDs
            try:
                # faiss_index.add_with_ids(embeddings.astype('float32'), ids)
                add_vectors(self.faiss_manager, self.uuid, embeddings.astype('float32'), ids)
                if debug: print("Made it to here 5")
            except Exception as e:
                print(f"Error adding to FAISS index: {e}")
                send_log_message(message=f"ERROR: FAILURE adding to FAISS index: {e}")
                raise

        # Cleanup
        cursor.close()
        self.connection_pool.putconn(conn)

    def load_memory_vector_db(self, memories_dict=None, debug = False):
        """
        Loads a dictionary of memory descriptions into a FAISS index for efficient similarity search.
        """

        # Start the timer so we can log how long this takes
        start_time = time.time()
        send_log_message(f"INFO: start memories FAISS DB load", _host=self.logging_host, _port=self.logging_port)

        if memories_dict is None:
            # Load full memories dict for agent if a memories_dict isn't passed in
            memories_dict = self.get_memories_dict()

        if debug: print(f"Full Memories Dict:",memories_dict,"\n\n")

        # Create a new dictionary with just the IDs and DESCRIPTIONS from the memories
        mem_descriptions_dict = {
            str(id): ' '.join([
                memory.get('description', '').strip(),
                memory.get('keywords', '').strip(),
                memory.get('primary_emotions', '').strip()
            ]).strip()
            for id, memory in memories_dict.items()
        }

        if debug: print("MEM Descriptions DICT:", mem_descriptions_dict,"\n\n")

        #Create a new list of just the memory descriptions (without ID keys)
        descriptions_list = list(mem_descriptions_dict.values())

        if debug: print("Descriptions LIST:", descriptions_list, "\n\n")

        #Get Embedding Dimension for the model we are using
        embedding_dimension = self.model.get_sentence_embedding_dimension()
        if debug: print(f"Embedding dimension1: {embedding_dimension}")

        # Check if GPU is available specificially for FAISS (This is different than CUDA available...)
        if gpu_check_avail_faiss():
            # Initialize FAISS index for GPU
            if debug: print("We are using CUDA for FAISS DB Index")
            send_log_message(message="INFO: Using CUDA for FAISS DB INDEX")
        else:
            # Initialize FAISS index for CPU
            if debug: print("Using FAISS CPU for FAISS DB Index")
            send_log_message(message="INFO: Using CPU for FAISS DB INDEX")

        try:
            # Attempt to sync the VectorDB Index with Postgres Memory Index
            # Note: create_faiss_index will check if the index already exists, and create it if it does not...
            create_faiss_index(self.faiss_manager, self.uuid, embedding_dimension)

            #Note: This is perhaps not as efficient as it needs to be, by forcing the recreate=True.
            #20240926MGY: The reason we have to force recreate is specificailly for EMHAT Experimentaiton.
            #   IE: If simulation memories are set to 'deactivated = 1' AFTER the experimentation is finished
            #   The only way to ensure that they are removed from the agent is to re-sync the agent's memory system
            #   We may need to reconsider this with Comp-HuSim Large Runs with 10K+ agents...

            self.sync_memory_vector_db(dimension=embedding_dimension,recreate=True)

        except Exception as e:
            print(f"Failed to initialize FAISS GPU index: {e}")
            send_log_message(message=f"ERROR: Using GPU for FAISS DB INDEX: {e}")
            raise

        if debug: print("Made it to here 3")

        # End the timer
        end_time = time.time()

        # Calculate and display elapsed time
        elapsed_time = end_time - start_time
        send_log_message(f"INFO: complete load FAISS DB elapsed time: {elapsed_time} seconds", _host=self.logging_host,
                         _port=self.logging_port)

        # Return the FAISS index object
        #return faiss_index

        return True

    def store_memory(self, activity_uuid, summary, model='gpt-4o-mini', debug=False):
        """
           Stores a memory generated from an activity summary into the database.

           This function performs the following steps:
           1. Sets up the database connection and cursor.
           2. Retrieves the activity name using the provided activity UUID.
           3. Constructs a prompt for the language model (LLM) to generate a memory based on the activity name and summary.
           4. Sends the prompt to the LLM and processes the response.
           5. Parses the LLM response into a dictionary.
           6. Stores the generated memory into the database along with additional metadata.
           7. Closes the cursor and returns the database connection to the connection pool.

           Parameters:
           activity_uuid (str): The unique identifier for the activity.
           summary (str): A summary of the event or activity.
           model (str, optional): The model to use for generating the memory. Default is 'gpt-4o'.

           Returns:
           None
           """

        # Get DB Stuff set up
        conn = self.connection_pool.getconn()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        activity_name = get_activity_name(cursor, activity_uuid=activity_uuid)

        #TODO: This function currently (20240607) only works with gpt-4o / gpt-3.5-turbo (LLAMAs fails to produce JSON)
        #model='gpt-4o-mini-2024-07-18'

        llm_prompt = f"""Imagine that YOU are the PERSON described by the PERSON TRAITS below. You have just now completed this ACTIVITY: {activity_name}.
                Here is a SUMMARY of the what happened:
                    [{summary}]

                The MEMORY of this EVENT needs to be stored in an efficient way for YOU to remember quickly in future.
                If this was a CONVERSATION about a past event, be sure to clarify in your memory that this was a CONVERSATION about a past event, not the memory of the past event itself.

                Based on your TRAITS (described below), please generate and return an ANSWER that is ONLY a JSON OBJECT with the following format.
                Do NOT include any other commentary in your anser. Structure your answer like this:

                {{
                    "memory_text": "<a first-person description of what happened in this event, refracted through this PERSONALITY>",
                    "memory_intensity": <floating point number from 0 to 1, with 0 being low intensity and 1 being VERY intense. Most memories should range from 0.3 to 0.5. Only particularly relevant or intense memories should be greater than 0.5>,
                    "memory_keywords": "<CSV string containing no more than three or four keywords that describe this memory>",
                    "memory_emotions": "<CSV string containing no more than three or four emotions associated with this memory>",
                    "memory_quality": <floating point number from -1 to 1, with -1 being VERY negative, 1 being VERY positive, and 0 being NEUTRAL>
                }}

                Your TRAITS are as follows:
                {self.get_traits()}
        """

        if debug:
            print(f"\n{llm_prompt}\n")

        # Process prompt with LLM
        gpt_response = get_llm_response(config=self.config, prompt=llm_prompt, model=model, client=self.client)

        if debug:
            print(f"\n{gpt_response}\n")

        # This is the MEMORY dictionary object
        memory_dict = json.loads(gpt_response)

        #Store this new memory dictionary object to the memories table
        new_memory_uuid = self.store_memory_to_db(activity_uuid, memory_dict, summary, llm_prompt, gpt_response, model, quality=0, agent_age=self.db_results["demographics"]['Age'])

        # Cleanup
        cursor.close()
        self.connection_pool.putconn(conn)

        return new_memory_uuid

    def assoc_agent_memories(self, memory_uuid, item_uuid, item_type='agent'):
        # Get DB Stuff set up
        conn = self.connection_pool.getconn()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Create the insert query to add the Memory to the DB
        insert_query = """
                        INSERT INTO memory_links (belongs_to,memory_uuid,item_id,item_type)
                        VALUES (%s, %s, %s, %s)
                        """

        # Values to insert into the DB
        values_to_insert = (self.uuid, memory_uuid, item_uuid, item_type)

        # Execute DB Query / Commit
        cursor.execute(insert_query, values_to_insert)
        conn.commit()

        # send_log_message(f"INFO: Created Memory Association: {self.uuid}, {memory_uuid}, {uuid_item}, {uuid_item_type}",_host=self.logging_host, _port=self.logging_port)

    def store_memory_to_db(self, activity_uuid, memory_dict, summary, llm_prompt, gpt_response, model, quality, agent_age, debug=False):
        """
           Stores a generated memory into the database and updates the FAISS index.

           This function performs the following steps:
           1. Sets up the database connection and cursor.
           2. Handles a null or empty activity UUID by setting it to None if necessary.
           3. Creates an SQL insert query to add a new memory record to the database.
           4. Defines the values to be inserted into the database, including memory details and metadata.
           5. Executes the insert query and commits the transaction.
           6. Retrieves the UUID of the newly inserted memory record.
           7. Converts the UUID into a 64-bit integer to be used as the FAISS vector database ID.
           8. Updates the newly inserted record with the 64-bit integer for the vector_db_id column.
           9. Reloads the FAISS memory object to include the new memory in the FAISS index.
           10. Closes the cursor and returns the database connection to the connection pool.

           Parameters:
           activity_uuid (str): The unique identifier for the activity. If empty, it is set to None.
           memory_dict (dict): Dictionary containing the memory details generated by the LLM.
           summary (str): A summary of the event or activity.
           llm_prompt (str): The prompt sent to the language model.
           gpt_response (str): The response received from the language model.
           model (str): The model used to generate the memory.
           quality (float): The quality score of the memory.
           agent_age (int): The age of the agent.

           Returns:
           None
           """

        # Get DB Stuff set up
        conn = self.connection_pool.getconn()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # handle null activity_uuid
        # Todo: Rearrange function call so that it is possible to overload activity_uuid as None in function call
        if activity_uuid == '':
            activity_uuid = None

        #Get the agent's current event UUID
        if self.current_event['agent_event_uuid'] is not None:
            current_event_uuid = self.current_event['agent_event_uuid']
        else:
            current_event_uuid = None

        #Handle override of activity_uuid as 00000000-0000-0000-0000-000000000000
        if activity_uuid == '00000000-0000-0000-0000-000000000000':
            current_event_uuid = '00000000-0000-0000-0000-000000000000'

        # Create the insert query to add the Memory to the DB
        insert_query = """
                INSERT INTO memories (belongs_to, related_action, keywords, primary_emotions, intensity, description, summary, prompt, response, model, quality, agent_age, agent_event_uuid)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING uuid
                """

        # Values to insert into the DB
        values_to_insert = (self.uuid, activity_uuid, memory_dict['memory_keywords'], memory_dict['memory_emotions'],
                            memory_dict['memory_intensity'], memory_dict['memory_text'], summary, llm_prompt,
                            gpt_response, model, memory_dict['memory_quality'], agent_age, current_event_uuid )

        # Execute DB Query / Commit
        cursor.execute(insert_query, values_to_insert)
        new_memory_uuid = cursor.fetchone()['uuid']
        conn.commit()

        # Convert UUID into Vector_DB_ID BIGINT
        vector_db_id = uuid_to_int64(new_memory_uuid)

        # Update the newly inserted record with the 64-bit integer for the vector_db_id column
        # This is the ID we will use for the FAISS ID
        update_query = """
            UPDATE memories
            SET vector_db_id = %s
            WHERE uuid = %s
        """
        cursor.execute(update_query, (vector_db_id, new_memory_uuid))
        conn.commit()

        # Create Description of Memory and Create Embedding for Memory
        # Create a new dictionary with just the IDs and DESCRIPTIONS from the memories
        mem_descriptions_dict = {
            str(vector_db_id): (
                    memory_dict['memory_text'] + ' ' +
                    memory_dict['memory_keywords'] + ' ' +
                    memory_dict['memory_emotions']
            ).strip()
        }

        if debug: print("NEW MEMORY: MEM Descriptions DICT:", mem_descriptions_dict, "\n\n")

        # Create a new list of just the memory descriptions (without ID keys)
        descriptions_list = list(mem_descriptions_dict.values())

        if debug: print("NEW MEMORY Descriptions LIST:", descriptions_list, "\n\n")

        # Convert descriptions to embeddings
        embeddings = self.model.encode(descriptions_list, convert_to_tensor=True)
        if debug: print("Completed embeddings")

        # Convert to numpy array
        embeddings = embeddings.cpu().detach().numpy()

        #Get IDs from mem_descriptions_dict
        ids = np.array([id for id in mem_descriptions_dict.keys()], dtype=np.int64)

        # Update the agent's memory objet
        add_vectors(self.faiss_manager, self.uuid, embeddings.astype('float32'), ids)

        # Cleanup
        cursor.close()
        self.connection_pool.putconn(conn)

        return new_memory_uuid

    def get_short_term_memories_prompt(self,event_context_uuid = None,debug=False):
        # Get DB Stuff set up
        conn = self.connection_pool.getconn()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Create list of SHORT-TERM json memories if event_context_uuid is set
        if event_context_uuid is None:
            # If the event_context_uuid was not passed in to the ask_question(), then we set the context to the agent's current agent_event_uuid
            # Ie: this allows us to have a simulacra of a working memory based on the currently engaged activity
            event_context_uuid = self.current_event['agent_event_uuid']

        if event_context_uuid is not None:
            # If the event_context_uuid is given, then we only want memories that are relevant to the event.
            # TODO: Incorporate LIST of event_context_uuids to allow for more than one context to be included.
            # event_context_uuid_list_str = ','.join(event_context_uuid)

            short_term_memory_prompt = "You do NOT have any SHORT TERM MEMORIES about this. Therefore, START AT THE BEGINNING by first making a PLAN. Then, start with the first item in your plan."
            memory_descriptions_short_term_json = ""

            if debug: print(f"USING THIS EVENT CONTEXT UUID TO PULL SHORT TERM MEMORIES: {event_context_uuid}")

            specific_event = get_agent_event_row(cursor = cursor, agent_event_uuid=event_context_uuid)
            relevant_short_term_memories = self.get_memories_dict(event_context_uuid=event_context_uuid, debug=debug)

            if debug:
                    print(f"Relevant Short Term Memories Dict: {relevant_short_term_memories}")

            # Create list of json memories
            if len(relevant_short_term_memories) > 0:
                memory_descriptions = [
                    {
                        "created_at": v['created_at'].strftime('%Y-%m-%d %H:%M:%S'),
                        "description": v['description'],
                        "primary_emotions": v['primary_emotions']
                    } for k, v in relevant_short_term_memories.items()
                ]
                # Serialize the descriptions dictionary to a JSON formatted string
                memory_descriptions_short_term_json = json.dumps(memory_descriptions, indent=4)

                # Display the relevant memories retrieved
                if debug: print(f"SHORT TERM RELEVANT MEMORIES JSON: ", memory_descriptions_short_term_json)

                # Construct the short-term memory prompt for inclusion below
                short_term_memory_prompt = f"""
                    As you answer this question, be particularly attentive of any SHORT-TERM PERSONAL MEMORIES listed below.
                    These SHORT-TERM MEMORIES will provide immediate context and information that you should keep in mind.
                    These memories are related to the following ACTIVITY:
                        {{ "ACTIVITY NAME": "{specific_event['activity_name']}", "REASON": "{specific_event['intention']}", "CONTEXT": "{specific_event['context_info']}" }}.
                        (NOTE: Do NOT HALLUCINATE additional context).
                    Notice the CREATED_AT timestamp, as this will provide the sequence of activities you have recently done.
                    The first memories listed are the MOST RECENT things you have done.
                    Be attentive to the DESCRIPTION of each memory.
                    GO SLOW. If you HAVE a SHORT TERM MEMORY about this thing, assume you have done that step.
                    However, if you do NOT have a SHORT TERM MEMORY about this thing, assume you have NOT done that step.
                    If you have been asked what you will do next, DO NOT KEEP REPEATING THE SAME THING AGAIN AND AGAIN. DO NOT GET STUCK IN A LOOP.

                    SHORT TERM MEMORY LIST: {memory_descriptions_short_term_json}
                """

            # Cleanup
            cursor.close()
            self.connection_pool.putconn(conn)

            return short_term_memory_prompt

    def get_long_term_memories_prompt(self,question,debug=False):
        long_term_memory_prompt = ""
        memory_descriptions_long_term_json = ""

        # LONG TERM via FAISS
        relevant_long_term_memories = self.faiss_remember(question, 20)

        # Create list of LONG-TERM json memories
        if relevant_long_term_memories is not None:
            memory_descriptions = [
                {
                    "created_at": v['created_at'].strftime('%Y-%m-%d %H:%M:%S'),
                    "description": v['description'],
                    "primary_emotions": v['primary_emotions']
                } for k, v in relevant_long_term_memories.items()
            ]
            # Serialize the descriptions dictionary to a JSON formatted string
            memory_descriptions_long_term_json = json.dumps(memory_descriptions, indent=4)

            # Display the relevant memories retrieved
            if debug: print(f"LONG TERM RELEVANT MEMORIES: ", memory_descriptions_long_term_json)

            long_term_memory_prompt = f"""
                        Finally, consider and be aware of LONG-TERM PERSONAL MEMORIES listed below as you answer this question.
                        GO SLOW. Refer to these LONG-TERM MEMORIES ONLY if these memories are especially relevant to the question or help you accomplish the task you are engaged in.
                        LONG TERM MEMORY LIST: {memory_descriptions_long_term_json}
                        """

        return long_term_memory_prompt

    def make_next_decision(self,model='gpt-4o-mini',environment='',debug=False,event_context_uuid=None,exclude_long_term_memories=False,store_memory=True):
        # Set up variables
        current_time = datetime.utcnow()
        short_term_memory_prompt = ""
        long_term_memory_prompt = ""
        environment_prompt=""

        # Get the Short Terrm Memories Prompt to add below
        short_term_memory_prompt = self.get_short_term_memories_prompt(event_context_uuid=event_context_uuid,debug=debug)
        if debug: print(f"SHORT TERM MEMORY PROMPT: {short_term_memory_prompt}")

        if environment != "":
            environment_prompt = f"Your immediate environment is described as: {environment}"

        # Only get this if we are NOT excluding long term memories
        if exclude_long_term_memories is False:
            #Extract Long-Term Relevant Topics to include:
            #There is probably a better way to do this:
            question = f"ACTIVITY: {self.current_event['activity_name']}. INTENTION: {self.current_event['intention']}"

            long_term_memory_prompt = self.get_long_term_memories_prompt(question,debug)  # Get Relevant Memories, if agent has memories
            if debug: print(f"LONG TERM MEMORY PROMPT: {long_term_memory_prompt}")

        # Generate the GPT Prompt to answer the question
        llm_prompt = f"""Imagine that you are the PERSON described by the TRAITS below.
                You are currently engaged in the following ACTIVITY: {self.current_event['activity_name']}.
                You are doing this ACTIVITY for this INTENTION: {self.current_event['intention']}
                The relevant context for this ACTIVITY is: {self.current_event['context_info']}
                The time is currently {current_time}.

                {environment_prompt}

                Based on the current ACTIVITY in which you are engaged, and oriented toward the INTENTION for that ACTIVITY, answer this question: "WHAT WILL YOU DO NEXT? WHY?"

                Your response should be VERY SHORT, and in the format of: {{"action":"<your next action>", "rationale":"<your justification for your answer>"}}.
                Do NOT provide any context or explanation in the ACTION section of your response. But DO provide context and explanation in the RATIONALE section.

                ANSWER in a FIRST PERSON, nuanced, appropriately complex way that fits the YOUR TRAITS as described below.

                Do NOT ramble about who you are. Use the TRAITS and MEMORIES as background to help shape your response.

                Allude to your TRAITS in your answer, but reference your TRAITS only when VERY relevant.

                YOUR TRAITS are as follows:
                    {self.get_traits()}

                {short_term_memory_prompt}

                {long_term_memory_prompt}
                """

        if debug: print(f"\n\nLLM Prompt:", llm_prompt)

        # Get the response from the LLM.
        gpt_response = get_llm_response(config=self.config, prompt=llm_prompt, model=model, client=self.client)
        answer = json.loads(gpt_response)['action']
        rationale = json.loads(gpt_response)['rationale']
        action_summary = answer + " " + rationale

        if debug: print(f"LLM Response:", gpt_response)

        #Store this memory to the DB for the Agent
        self.summarize_and_store_action_memory(action_summary=action_summary,debug=debug)

        return gpt_response


    def ask_question(self, question, model='gpt-40-1', debug=False, event_context_uuid = None, exclude_long_term_memories=False, response_length=None, remember_interaction=True):
        # Ask a question of the agent.

        # Set up variables
        current_time = datetime.utcnow()
        short_term_memory_prompt = ""
        long_term_memory_prompt = ""

        # Get an LLM predicted length of the response if response_length was not provided
        if response_length is None:
            predicted_length = get_predicted_length(self.config, question)
        else:
            predicted_length = response_length

        # Get the Short Terrm Memories Prompt to add below
        short_term_memory_prompt = self.get_short_term_memories_prompt(event_context_uuid=event_context_uuid, debug=debug)
        if debug: print(f"SHORT TERM MEMORY PROMPT: {short_term_memory_prompt}")

        #Only get this if we are NOT excluding long term memories
        if exclude_long_term_memories is False:
            long_term_memory_prompt = self.get_long_term_memories_prompt(question,debug)   # Get Relevant Memories, if agent has memories
            if debug: print(f"LONG TERM MEMORY PROMPT: {long_term_memory_prompt}")

        # Generate the GPT Prompt to answer the question
        llm_prompt = f"""Imagine that you are the PERSON described by the TRAITS below.
        You are currently engaged in the following ACTIVITY: {self.current_event['activity_name']} for this reason: {self.current_event['intention']}. With this CONTEXT: {self.current_event['context_info']}. (NOTE: Do NOT HALLUCINATE additional context).
        The time is currently {current_time}.
        You have just been asked this QUESTION: ("{question}").
        Your answer should be of {predicted_length} length.

        ANSWER in a FIRST PERSON, nuanced, appropriately complex way that fits the PERSON's TRAITS.

        Do NOT ramble about who you are. Use the TRAITS and MEMORIES as background to help shape your response.

        Integrate references to your TRAITS into your response ONLY WHERE RELEVANT for MEDIUM or LONG responses.

        Your TRAITS are as follows:
            {self.get_traits()}

        {short_term_memory_prompt}

        {long_term_memory_prompt}
        """

        if debug: print(f"\n\nLLM Prompt:", llm_prompt)

        #Get the response from the LLM.
        gpt_response = get_llm_response(config=self.config, prompt=llm_prompt, model=model, client=self.client)

        if debug: print(f"LLM Response:", gpt_response)

        #TODO: This needs to be updated so that it supports multiprocessing / multithreading for efficiency
        #TODO: Run the summarize_and_store process in the background so that we can get the results back to the calling process ASAP.
        #CHALLENGE: As of 20240608 This isn't working because I can't run a subprocess with the db_connection_pool as implemented....
        #SOLUTION: TBD
        #background_process = multiprocessing.Process(target=self.summarize_and_store_interaction, args=(question, gpt_response, llm_prompt, model))
        #background_process.start()

        if remember_interaction is True:
            self.summarize_and_store_question_and_response_memory(question=question, response=gpt_response, llm_prompt = llm_prompt, model=model, debug=debug)

        return gpt_response

    def summarize_and_store_action_memory(self, action_summary='', model='gpt-4o-mini', debug=False):
        #Summarizes and stores the memory of an action for an agent

        #Setup DB Connections
        conn = self.connection_pool.getconn()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        if action_summary == '':
            send_log_message("ERROR: Action Summary was empty. Cannot store memory.")
            print("ERROR: Action Summary was empty. Cannot store memory.")
            return 0

        # Construct Summarization prompt
        summarization_prompt = f"""Extract a VERY brief FIRST PERSON summary of the following ACTIVITY that you just COMPLETED. This summary will be stored as a memory for your future use.
        Here is what you were doing and why: {action_summary}
        Assume that this action is now in the past.
        """

        if debug:
            print(f"Summarization prompt: {summarization_prompt}")

        #Get summary of question and response
        summary = get_llm_response(config=self.config, prompt=summarization_prompt, model=model, client=self.client)

        if debug:
            print(f"Summary of memory: {summary}")

        #TODO: Do we need these?
        #Record the interaction (ie: the 3rd person omniscent history of what happened)
        #self.insert_interaction(question, model, response, llm_prompt, summary)

        #Find the activity name for this activity
        #activity_uuid = find_activity(activity_description=summary, model='gpt-4o-mini', cursor=cursor,config=self.config, create_activity = True)
        # TODO:
        # NOTE: THis is only relevant for the EMHAT scenario that is the "Search for the injured"
        activity_uuid = 'b73d118e-be77-4971-846c-b261db8d2972'          #IE: Activity: "Search for the injured"
        if debug: print(f"Summarize and Store() Activity UUID:",activity_uuid)

        # Store the memory in the DB
        new_memory_uuid = self.store_memory(activity_uuid = activity_uuid, summary=summary, model=model, debug=debug)

        # Clean Up & return DB connection to pool
        cursor.close()
        self.connection_pool.putconn(conn)

        return new_memory_uuid

    def summarize_and_store_question_and_response_memory(self, question, response, llm_prompt='', model='gpt-4o', debug=False):
        #Setup DB Connections
        conn = self.connection_pool.getconn()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Construct Summarization prompt
        summarization_prompt = f"""Extract a VERY brief summary of the following CONVERSATION you just had, for use as memory and context in future.
        Be sure to make it clear that this was a CONVERSATION. Do not use 2nd person voice, but rather first person voice. For example: \'I recently had a conversation. Someone said XYZ to me. I responded by saying ABC\'.
        Here is what was said in the conversation:
        Someone SAID THIS TO YOU: {question}
        YOUR RESPONSE WAS: {response}
        """

        if debug:
            print(f"Summarization prompt: {summarization_prompt}")

        #Get summary of question and response
        summary = get_llm_response(config=self.config, prompt=summarization_prompt, model=model, client=self.client)

        if debug:
            print(f"Summary of memory: {summary}")

        # Record the interaction (ie: the 3rd person omniscent history of what happened)
        self.insert_interaction(question, model, response, llm_prompt, summary)

        #Find the activity name for this activity
        activity_uuid = find_activity(activity_description='Answering a verbal question', cursor=cursor,config=self.config, create_activity = True)

        if debug:
            print(f"Summarize and Store() Activity UUID:",activity_uuid)

        # Store the memory in the DB
        new_memory_uuid = self.store_memory(activity_uuid = activity_uuid, summary=summary, model=model, debug=debug)

        # Clean Up & return DB connection to pool
        cursor.close()
        self.connection_pool.putconn(conn)

        return new_memory_uuid

    def insert_interaction(self, question, llm_used, gpt_response, prompt, summary):
        conn = self.connection_pool.getconn()
        cursor = conn.cursor()

        insert_query = """
        INSERT INTO interactions (agent_uuid, question, llm_used, response, prompt, summary)
        VALUES (%s, %s, %s, %s, %s,%s)
        """

        # Execute DB Commit and clean up
        cursor.execute(insert_query, (self.uuid, question, llm_used, gpt_response, prompt, summary))
        conn.commit()
        cursor.close()
        self.connection_pool.putconn(conn)

    def summarize(self):
        """
        Return the agent's summary
        """
        # return get_agent_summary(self.uuid,self.cursor)
        return self.db_results['summary']

    def get_uuid(self):
        """
        Return the UUID for this agent (ie: self)
        :return: UUID for self
        """
        return self.uuid

    def create_agent_uuid(self):
        # Create an agent in the DB without any additional informaiton other than UUID

        # Create SQL Query
        query = sql.SQL("""
                INSERT INTO agents DEFAULT VALUES
                RETURNING uuid
                """)

        # print (query)

        # ToDo: Put the below into its own function: self.execute_auery(). Figure out how to handle the logging.
        # ToDo: Also bring this in from the create() function

        # Get Cursor and Connection to DB
        # Attempt to update DB with Query.
        # Handle errors gracefully
        conn = self.connection_pool.getconn()
        try:
            cursor = conn.cursor()
            # Execute SQL Query
            cursor.execute(query)
            conn.commit()

            # Get the new UUID for the new agent
            new_uuid = cursor.fetchone()[0]
            self.uuid = new_uuid
            send_log_message(f"INFO: CREATED EMPTY AGENT {self.uuid}",_host=self.logging_host, _port=self.logging_port)
            # send_log_message(f"INFO: LLM RESPONSE: {data}",_host=self.logging_host, _port=self.logging_port)

        except Exception as e:
            print(f"DB Error occurred")
            send_log_message(f"ERROR: DB Error {e}",_host=self.logging_host, _port=self.logging_port)
            conn.rollback()

        finally:
            cursor.close()
            self.connection_pool.putconn(conn)

    def update_db_column(self, column_name, column_value):
        # Update the DB so that this agent is rendered Visible / Invisible (for UX display purposes)
        # Create SQL Query
        query = sql.SQL("""
            UPDATE agents
            SET {column} = {value}
            WHERE uuid = {uuid}
        """).format(column=sql.Identifier(column_name),
                    value=sql.Literal(column_value),
                    uuid=sql.Literal(self.uuid))

        # print (query)

        # ToDo: Put the below into its own function: self.execute_auery(). Figure out how to handle the logging.
        # ToDo: Also bring this in from the create() function

        # Get Cursor and Connection to DB
        # Attempt to update DB with Query.
        # Handle errors gracefully
        conn = self.connection_pool.getconn()
        try:
            cursor = conn.cursor()
            # Execute SQL Query
            cursor.execute(query, (column_name, column_value, self.uuid))
            conn.commit()

            # Log changes
            send_log_message(f"INFO: UPDATED AGENT DB. UUID: {self.uuid}",_host=self.logging_host, _port=self.logging_port)

        except Exception as e:
            print(f"DB Error occurred")
            send_log_message(f"ERROR: DB Error {e}",_host=self.logging_host, _port=self.logging_port)
            conn.rollback()

        finally:
            # If we were able to update the DB, then we need to re-load the agent's details into the object
            self.load_agent_details()
            cursor.close()
            self.connection_pool.putconn(conn)

    def change_visibility(self, visibility=0):
        # Update the DB so that this agent is rendered Visible / Invisible (for UX display purposes)
        # Create SQL Query
        query = """
            UPDATE agents
            SET visible = %s
            WHERE uuid = %s
        """

        # print (query)

        # ToDo: Put the below into its own function: self.execute_auery(). Figure out how to handle the logging.
        # ToDo: Also bring this in from the create() function

        # Get Cursor and Connection to DB
        # Attempt to update DB with Query.
        # Handle errors gracefully
        conn = self.connection_pool.getconn()
        try:
            cursor = conn.cursor()
            # Execute SQL Query
            cursor.execute(query, (visibility, self.uuid))
            conn.commit()

            # Log changes
            send_log_message(f"INFO: UPDATED AGENT VISIBILITY. UUID: {self.uuid}",_host=self.logging_host, _port=self.logging_port)

        except Exception as e:
            print(f"DB Error occurred")
            send_log_message(f"ERROR: DB Error{e}",_host=self.logging_host, _port=self.logging_port)
            conn.rollback()

        finally:
            cursor.close()
            self.connection_pool.putconn(conn)

    def create(self):
        # TODO: Comment this function

        creation_file = 'data/creation_prompt.txt'  # NOTE: this is a static file, but the "SEED" at the end can be randomly generated to add variety.

        # open the file in read mode and store content in a variable
        with open(creation_file, 'r') as file:
            content = file.read()

        # Get the seed for the Agent genreation LLM prompt
        seed = generate_persona_seed()

        # print(f"SEED:",seed)

        creation_prompt = "Based on the SEED below and utilizing the given PERSONA structure below, generate a creative and unique persona, " \
                          "blending uncommon and unrelated fields or interests and avoiding common career choices. (But please, something besides puppetry)." \
                          "Format all the generated information into one single JSON object based on the PERSONA structure: " + seed + content

        send_log_message(f"======================",_host=self.logging_host, _port=self.logging_port)
        send_log_message(f"INFO: CREATING: AGENT",_host=self.logging_host, _port=self.logging_port)
        #send_log_message(f"INFO: CREATION PROMPT: " + creation_prompt, _host=self.logging_host, _port=self.logging_port)
        send_log_message(f"======================",_host=self.logging_host, _port=self.logging_port)

        response = get_llm_response(config=self.config, prompt=creation_prompt, model='gpt-4o', client=self.client)

        #print(response)

        data = json.loads(response)

        # print(data)

        # Extract details from JSON response from LLM
        first_name = data["PERSONA"]["Name"]["First"]
        last_name = data["PERSONA"]["Name"]["Last"]
        backstory = data["PERSONA"]["Backstory"]
        hook = data["PERSONA"]["Hook"]
        summary = data["PERSONA"]["Summary"]
        profile_pic_description = data["PERSONA"]["Profile_Picture_Description"]
        demographics = json.dumps(data["PERSONA"]["Demographics"])
        personality = json.dumps(data["PERSONA"]["Personality"])
        psychographics = json.dumps(data["PERSONA"]["Psychographics"])
        interests = json.dumps(data["PERSONA"]["Interests"])
        family = json.dumps(data["PERSONA"]["Family"])

        # Create SQL Query
        query = sql.SQL("""
        INSERT INTO agents (
            first_name, last_name, backstory, hook,
            profile_pic_description, demographics, psychographics,
            personality, summary, interests, family
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING uuid
        """)

        # print (query)

        # Get Cursor and Connection to DB
        # Attempt to update DB with Query.
        # Handle errors gracefully
        conn = self.connection_pool.getconn()

        try:
            cursor = conn.cursor()
            # Execute SQL Query
            cursor.execute(query, (
                first_name, last_name, backstory, hook,
                profile_pic_description, demographics, psychographics,
                personality, summary, interests, family
            ))
            conn.commit()

            # Get the new UUID for the new agent
            new_uuid = cursor.fetchone()[0]
            self.uuid = new_uuid
            send_log_message(f"INFO: CREATED AGENT {self.uuid}",_host=self.logging_host, _port=self.logging_port)
            # send_log_message(f"INFO: LLM RESPONSE:"+data,_host=self.logging_host, _port=self.logging_port)
            self.db_results = get_agent_row(uuid=self.uuid,cursor=cursor)


            print(f"Created new agent: {self.uuid}")

        except Exception as e:
            print(f"DB Error occurred")
            send_log_message(f"ERROR: DB Error {e}",_host=self.logging_host, _port=self.logging_port)
            conn.rollback()
        finally:
            cursor.close()
            self.connection_pool.putconn(conn)

    def dream(self, topic=get_topic_random(), dream_quality=random.choices(["GOOD", "BAD", "TRANQUIL", "SCARY"]), do_random=False, model='gpt-4o', do_print=False):
        """
        This function consolidates and summarizes the agent's experiences for the day, and stores them into the agent's memory network.
        A random memory from the day is selected and this memory becomes a part of the "REM" stage of sleep for the agent.
            This REM stage of sleep for the agent is then stored as a dream-memory in the agent's memory network.
            Furthermore, the 'intensity' value of that original memory is increased in the agent's memory network.
        :return: dict with information regarding the dreaming process
        """

        # Setup DB Connections
        conn = self.connection_pool.getconn()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Get Random / Relevant Memories, if agent has memories
        if self.memories_dict:
            if do_random:
                relevant_memories = self.get_random_memory()
            else:
                relevant_memories = self.faiss_remember(topic, nearest_neighbors=2, llm_check_memories=False)

            # Create list of json memories
            memory_descriptions = [
                {
                    "created_at": v['created_at'].strftime('%Y-%m-%d %H:%M:%S'),
                    "description": v['description'],
                    "keywords": v['keywords'],
                    "primary_emotions": v['primary_emotions']
                } for k, v in relevant_memories.items()
            ]
            # Convert memory descriptions to a JSON formatted string
            memory_descriptions_json = json.dumps(memory_descriptions, indent=4)
        else:
            memory_descriptions_json = ''

        # Un-Comment this to display the relevant memories retrieved
        # print(memory_descriptions_json)

        # Generate the GPT "Dream" Prompt
        llm_prompt = f"""Imagine that you are the PERSON described by the TRAITS below.
                        You you are dreaming a {dream_quality} dream about this topic: {topic}
                        Describe the dream.
                        Be specific in what you see and feel and hear.
                        Make your answer of MEDIUM length.
                        Be aware of the MEMORIES listed below as you answer this question, but only reference those MEMORIES that are relevant to your DREAM.
                        ANSWER in a FIRST PERSON, nuanced, appropriately complex way that fits the PERSON TRAITS.
                        Do NOT ramble about who you are. However, use the PERSON TRAITS as background to help inform your response.

                        Integrate your TRAITS into your DREAM ONLY WHERE RELEVANT.

                        Your TRAITS are as follows: {self.get_traits()}

                        MEMORIES: {memory_descriptions_json}"""

        if do_print: print(f"LLM Prompt: ",llm_prompt)

        #Get Response from LLM
        gpt_response = get_llm_response(config=self.config, prompt=llm_prompt, model=model, client=self.client)

        if do_print: print(f"LLM Response: ",gpt_response)

        # Construct Summarization prompt
        summarization_prompt = f"""
           Extract a VERY brief FIRST PERSON summary of the following dream for use as memory and context in the future:
           Your DREAM: {gpt_response} """

        if do_print: print(f"Summarization Prompt: ",summarization_prompt)

        #Generate Summary of dream
        summary = get_llm_response(config=self.config, prompt=summarization_prompt, model='gpt-3.5-turbo', client=self.client)

        if do_print: print(f"LLM Summary: ",summary)

        #Get Activity ID for dreaming
        dream_activity_uuid = find_activity(activity_description='Dreaming', cursor=cursor,config=self.config, create_activity=True)

        #Store the memory of the dream
        self.store_memory(activity_uuid = dream_activity_uuid, summary=summary)

        # Clean Up & return DB connection to pool
        cursor.close()
        self.connection_pool.putconn(conn)

        return gpt_response

    def get_agent_details(self):
        return self.db_results

    def load_agent_details(self):
        """
        Retrieves and loads the details of an agent based on its UUID from the database.

        This function queries the 'agents' table in the database using the agent's UUID
        to fetch its details. The fetched details are then stored in the agent object's
        'db_results' attribute as a dictionary. The loading process is logged at both
        the start and the end.

        :attribute uuid: The UUID of the agent for which details are to be fetched.
                        This attribute should exist in the object invoking this method.
        :attribute real_dict_cursor: A psycopg2 RealDictCursor object used to execute
                                    SQL queries and fetch results in dictionary format.
        :attribute db_results: Dictionary storing the fetched details of the agent.
                                It's populated by this method.
        :attribute logger: Logger object for logging the details loading process.

        Note:
        Ensure that the 'uuid' attribute is set AND VALID before calling this method.
        Additionally, ensure the database connection and 'real_dict_cursor' are
        properly initialized.
        """
        start_time = time.time()

        # Calculate and display elapsed time
        send_log_message(message=f"INFO: ====STARTING AGENT LOADING====",_host=self.logging_host, _port=self.logging_port)
        send_log_message(message=f"INFO: UUID {self.uuid}",_host=self.logging_host, _port=self.logging_port)

        # Query DB for agent details
        query = sql.SQL("SELECT * FROM agents WHERE uuid = %s LIMIT 1;")
        conn = self.connection_pool.getconn()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(query, [self.uuid])

        # Get results from the cursor object and store dictionary into CompHuSim Agent Object
        self.db_results = cursor.fetchall()[0]

        #calculate sub-elapsed time and log.
        sub_time = time.time()
        elapsed_time = sub_time - start_time
        send_log_message(message=f"INFO: complete agent details load. UUID: {self.uuid}. Elapsed Time: {elapsed_time:.5g}", _host=self.logging_host, _port=self.logging_port)

        #Start loading memories
        send_log_message(message=f"INFO: start load agent memories.",_host=self.logging_host, _port=self.logging_port)

        # Load Memories for Agent
        #self.load_faiss_memories_object()  Note: This function was redundant and thus removed: MGY: 20240724
        self.load_memory_vector_db()

        # calculate sub-elapsed time and log.
        sub_time2 = time.time()
        elapsed_time2 = sub_time2 - sub_time

        send_log_message(message=f"INFO: complete load agent memories. Elapsed Time: {elapsed_time2:.5g} ",_host=self.logging_host, _port=self.logging_port)

        # print(self.db_results)
        # Log events
        send_log_message(message="INFO: ======AGENT LOADING COMPLETE====",_host=self.logging_host, _port=self.logging_port)

        # Clean Up
        cursor.close()
        self.connection_pool.putconn(conn)

    def get_demographics(self):
        return self.db_results['demographics']

    def get_name(self):
        return self.db_results['first_name'] + ' ' + self.db_results['last_name']

    def get_psychographics(self):
        return self.db_results['psychographics']

    def get_personality(self):
        return self.db_results['personality']

    def get_interests(self):
        return self.db_results['interests']

    def get_family(self):
        return self.db_results['family']

    def deactivate_event_memories(self,agent_event_uuid=None, debug=False):
        # Setup DB Connections
        conn = self.connection_pool.getconn()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        #Deactivate Agent Event Memories
        deactivate_agent_event_memories(cursor, agent_uuid=self.uuid, agent_event_uuid=agent_event_uuid, debug=debug)
        self.load_memory_vector_db()

        # Clean Up & return DB connection to pool
        cursor.close()
        self.connection_pool.putconn(conn)

    def deactivate_all_except_backstory_memories(self, debug=False):
        #Deactivates ALL EXCEPT backstory memories.
        # USE CAREFULLY!!!
        # Setup DB Connections
        conn = self.connection_pool.getconn()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        query = f"UPDATE MEMORIES set deactivated = 1 where belongs_to = '{self.uuid}' and agent_event_uuid <> '00000000-0000-0000-0000-000000000000'"
        cursor.execute(query)
        rows_affected = cursor.rowcount

        if(debug):
            print(f"Num Memories Deactivated: {rows_affected}")

        #Reload Vector DB
        self.load_memory_vector_db()

        # Clean Up
        conn.commit()
        cursor.close()
        self.connection_pool.putconn(conn)

    def reactivate_event_memories(self, agent_event_uuid=None, debug=False):
        # Setup DB Connections
        conn = self.connection_pool.getconn()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        #Reactivate Agent Event Memories
        reactivate_agent_event_memories(cursor, agent_uuid=self.uuid, agent_event_uuid=agent_event_uuid, debug=debug)
        self.load_memory_vector_db()

        # Clean Up & return DB connection to pool
        cursor.close()
        self.connection_pool.putconn(conn)

    def generate_backstory_memories(self, topic=None, do_random = False, age=None, num_memories=1, do_print=True,model='gpt-40-1'):

        # Get a random topic to build the memory form
        # ToDo: Expand this list to include more categories of memory to construct.
        # topic_list = ['friend', 'family', 'interest', 'hobby', 'mother', 'father', 'sibling', 'grandmother', 'grandfather', 'bully', 'school', 'holiday']

        if age == None:
            # Get a random age to build the memory from
            agent_age = int(self.db_results['demographics']['Age'])
            age = random.randint(3, agent_age)

        if topic == None or do_random == True:
            # Get a random topic to build the memory from
            do_random = True
            topic = get_topic_random()

        #Generate Intensity & Quality values for the memory: Two decimal places.
        memory_intensity = round(random.uniform(0.01, 1), 2)
        memory_quality = round(random.uniform(-1, 1), 2)

        backstory_memories_prompt = f"""Based on the TRAITS (ie: SUMMARY, DEMOGRAHPICS, PERSONALITY, PSYCHOGRAPHICS, INTERESTS, FAMILY) of the PERSONA described below,
                        generate ONE detailed description for a memory about this TOPIC: "{topic}" at age {age}, with the memory intensity {memory_intensity} (with zero being a low intensity memory and 1 being a VERY intense memory)
                        and a memory quality {memory_quality} (with -1 being a VERY negative memory, 1 being a VERY positive memory and 0 being a NEUTRAL memory).
                        The memory should be resonant with someone who has the charactaristics listed below.
                        Be specific in the sensory experiences, the people, the places, and the activities included in the memory.
                        Be especially attentive to the PERSONALITY and PSYCHOGRAPHICS of the persona.
                        Format all the generated information into one single JSON object based on this MEMORY structure:

                            - agent_age: <{age}>
                            - memory_topic: <{topic}>
                            - memory_text: <a first-person description of what happened in this event, refracted through this PERSONALITY>
                            - memory_keywords: <list of no more than three or four keywords as CSV string that describe what this memory is about>
                            - memory_intensity: <{memory_intensity}>
                            - memory_emotions: <list of no more than three or four emotions as CSV string associated with this memory>
                            - memory_summary: <brief summary of the memory, under 100 words>
                            - memory_quality: <{memory_quality}>

                        Your TRAITS are as follows: {self.get_traits()}
        """

        send_log_message(f"INFO: CREATING: BACKSTORY MEMORIES FOR AGENT {self.uuid}",_host=self.logging_host, _port=self.logging_port)
        #send_log_message(f"INFO: PROMPT: " + backstory_memories_prompt,_host=self.logging_host, _port=self.logging_port)

        #model = 'gpt-4o'
        gpt_response = get_llm_response(config=self.config, prompt=backstory_memories_prompt, model=model, client=self.client)

        if do_print:
            print(gpt_response)

        # Strip off leading characters that are returned by GPT4+ to designate a JSON object
        json_string = gpt_response.strip('```json\n')
        json_string = json_string.strip('```')
        memory_dict = json.loads(json_string)

        # print(memory_dict)

        # return gpt_response_json

        self.store_memory_to_db('00000000-0000-0000-0000-000000000000', memory_dict, memory_dict['memory_summary'], backstory_memories_prompt, gpt_response,
                                model, memory_dict['memory_quality'], memory_dict['agent_age'])

        #If we're generating more than one memory, recursively call this function.
        if num_memories > 1:
            next_run_count = num_memories -1
            self.generate_backstory_memories(num_memories=next_run_count,do_print=do_print,topic=topic,do_random=do_random)

    def get_random_memory(self):
        conn = self.connection_pool.getconn()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Get a random memory_id of self.uuid
        query = """ SELECT vector_db_id, uuid, description, summary, keywords, intensity, primary_emotions, created_at
                    from memories
                    where belongs_to = %s
                    ORDER BY RANDOM()
                    LIMIT 1;
        """
        cursor.execute(query, [self.uuid])
        rand_memory = cursor.fetchall()[0]
        cursor.close()
        memories_dict = {}

        memories_dict[rand_memory['vector_db_id']] = {
                    'uuid': rand_memory['uuid'],
                    'summary': rand_memory['summary'],
                    'description': rand_memory['description'],
                    'created_at': rand_memory['created_at'],
                    'keywords': rand_memory['keywords'],
                    'intensity': rand_memory['intensity'],
                    'primary_emotions': rand_memory['primary_emotions']
        # Clean Up
        }

        self.connection_pool.putconn(conn)

        # agent_list = []
        # agent_list.append()
        # return self.get_memories_dict(rand_memory['uuid'])
        return memories_dict

    def generate_profile_pic(self, num_images=1, engine_id ='sdxl'):
        """
        Args:
            num_images: int
            engine_id: str
                --> sdxl  (this is for SDXL v1.0)
                --> sd (this is for SD v1.5)
                --> controlnet-sdxl (this is for ControlNet SDXL)

        Returns:
            None but the generated image(s) would be saved in the "profile_pic" directory
        """

        OCTOAI_TOKEN = self.octoml_api_key

        url = f"https://image.octoai.run/generate/{engine_id}"
        profile_pic_description = self.db_results["profile_pic_description"]
        payload = {
            "prompt": f"A colored pixar-style art of {profile_pic_description}",
            "negative_prompt": "Blurry photo, distortion, low-res, bad quality",
            "steps": 30,
            "width": 1024,
            "height": 1024,
            "num_images": num_images,
        }

        headers = {
            "Authorization": f"Bearer {OCTOAI_TOKEN}",
            "Content-Type": "application/json",
        }

        response = requests.post(url, headers=headers, json=payload)

        if response.status_code != 200:
            print(response.text)

        img_list = response.json()["images"]

        for i, img_info in enumerate(img_list):
            img_bytes = base64.b64decode(img_info["image_b64"])
            img = PIL.Image.open(io.BytesIO(img_bytes))
            img.load()
            img.save(f"profile_pic/{self.uuid}_{i}.png")


        print(f"{num_images} image(s) generated sucessfully!")

    def get_last_memory_id(self):
        """

        Returns: the most recent memory_id

        """
        # Get DB Stuff set up
        conn = self.connection_pool.getconn()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Create the  query
        query = """
                    SELECT uuid FROM memories
                    WHERE belongs_to = %s
                    ORDER BY created_at DESC
                    LIMIT 1
                    """
        cursor.execute(query, [self.uuid])
        memory_id = cursor.fetchone()['uuid']

        # Cleanup
        cursor.close()
        self.connection_pool.putconn(conn)

        return memory_id

    def get_memory_decay_info(self):
        """

        Returns: rows of decayed memories belonging to the agent

        """
        pass
    def get_frequency(self, memory_uuid=''):
        """

        Args:
            memory_uuid: string or None

        Returns:
             -- empty string or None, return the latest memory frequency
             -- not empty string nor None, return the frequency corresponding to the memory_uuid
        """
        pass

    def get_strength(self):
        """

       Args:
           memory_uuid: string or None

       Returns:
            -- empty string or None, return the latest memory strength
            -- not empty string nor None, return the strength corresponding to the memory_uuid
       """

        pass

    def get_memory_decay(self):
        pass

    def forget(self, memory_uuid=''):
        """

        Args:
            memory: string or None
                -- empty string or None: forget random pieces of memory
                -- not empty string nor None: forget the specified memories (completely forget)

        Returns:
            nothing but update the memory_decay table
            TODO: update memory_links, memories and all other relevant tables
                to optimize memory storage once the forgetting mechanism is well implemented.
        """
        pass

    def insert_agent_decayed_memories(self, memory_uuid):
        pass

    def shutdown(self):
        if self.memories_dict and hasattr(self.faiss_memories, 'reset'):
            send_log_message(f"INFO: UNLOADING AGENT FROM MEMORY {self.uuid}",_host=self.logging_host, _port=self.logging_port)
            print("UNLOADING AGENT MEMORY " + self.uuid)
            self.faiss_memories.reset()
        gc.collect()
        torch.cuda.empty_cache()
