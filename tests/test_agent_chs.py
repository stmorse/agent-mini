import json
import time
import uuid

from psycopg2 import sql
from psycopg2.extras import RealDictCursor

from sentence_transformers import SentenceTransformer

from utilities.utils import generate_persona_seed, get_llm_response, get_agent_row
from utilities.logging_utils import send_log_message
from utilities.faiss_utils import get_faiss_manager

class CompHuSimAgent:
    def __init__(self, config=None, logger=None, connection_pool=None, 
                 this_uuid='', do_create=True):
        
        # set up stuff: connections to LOGGING SERVER and FAISS DB SERVER
        # needs config to be output of load_config()
        self.config = config
        self.logging_host = config.get('DEFAULT', 'LOGGING_HOST')
        self.logging_port = config.get('DEFAULT', 'LOGGING_PORT')
        self.faiss_host = config.get('DEFAULT', 'FAISS_HOST')
        self.faiss_port = config.get('DEFAULT', 'FAISS_PORT')
        self.faiss_manager = get_faiss_manager(
            a=(self.faiss_host, int(self.faiss_port)), key=b'faiss')

        # Memory span for agent
        # how many memories to recall when "remembering" from FAISS DB 
        # (ie: k nearest neighbors)
        self.memory_span = 10  

        # Set up the current activity that the agent is engaged in.
        self.nullify_current_event()

        # Set up the current task list that the agent maintas.
        self.task_list = None

        # Embedding model to use for agent's memory
        # This is hot-swappable....
        # See: https://www.sbert.net/docs/pretrained_models.html
        # self.embedding_model_name = 'all-MiniLM-L6-v2'
        self.embedding_model_name = 'all-mpnet-base-v2'
        self.model = SentenceTransformer(self.embedding_model_name)

        # Setup Connection Pool to DB. Exit if fail.
        self.connection_pool = connection_pool
        if not self.connection_pool:
            print("Failed DB Connection. Exiting")
            self._log(f"ERROR: FAILED STARTUP: Error connecting \
                             to database. SHUTDOWN.")
            exit()

        # if a UUID is passed, then load this agent with those details from DB
        # if no UUID is passed, but do_create is True, then create a new agent using persona template
        # else spin up a new agent using default values
        self.uuid = ''
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
                self._log(f"ERROR: FAILED TO LOAD AGENT: {e}")
                self.uuid = ''
        elif do_create:
            self.create(defaults=False)
            self.load_agent_details()
        else:
            # creates new agent with default values and saves self.uuid
            self.create(defaults=True)
            print(f"Created empty agent. UUID:", self.uuid)
            self.load_agent_details()

        # previous CHS agent used azure client
        # self.client = create_azure_client(config=self.config, model='gpt-40-1')
        # TODO: adapt for internal LLM server
        self.client = None

    """
    Internal utility functions
    """    

    def _log(self, message):
        # convenience method for logging
        send_log_message(message=message, 
                         _host=self.logging_host, 
                         _port=self.logging_port)

    def _execute_sql_query(self, query, values=None, fetch='all',
                           cf=None):
        # query should be a psycopg2.sql.SQL object
        # values should be a tuple of values to insert into the query
        # returns value from cursor.fetch{fetch}()[0]

        # connect to DB and execute query
        conn = self.connection_pool.getconn()
        result = None
        try:
            # execute SQL Query
            cursor = conn.cursor(cursor_factory=cf)
            cursor.execute(query, vars=values)
            conn.commit()

            if fetch=='all':
                result = cursor.fetchall()
            elif fetch=='one':
                result = cursor.fetchone()

        except Exception as e:
            print(f"DB Error occurred")
            self._log(f"ERROR: DB Error {e}")
            conn.rollback()

        finally:
            cursor.close()
            self.connection_pool.putconn(conn)

        return result

    """
    AGENT CREATION
    """

    def create(self, defaults=False):
        # Creates an agent in the DB
        # uses an LLM with the Persona template unless defaults=True

        self._log(f"======================")
        self._log(f"INFO: CREATING: AGENT")

        # if defaults is True, then create an agent with default values
        if defaults:
            # Create SQL Query
            # this inserts a row with default values and returns the UUID
            # of the newly created row
            query = sql.SQL("""
                    INSERT INTO agents DEFAULT VALUES
                    RETURNING uuid
                    """)

            res = self._execute_sql_query(query, fetch='one')
            self.uuid = res[0]
            self._log(f"INFO: CREATED EMPTY AGENT {self.uuid}")

            return

        # otherwise, use the persona template

        # static template -- random LLM variation and SEED ensure variation
        creation_file = 'data/persona_template.txt'  
        with open(creation_file, 'r') as file:
            content = file.read()

        # Get the seed for the Agent generation LLM prompt
        seed = generate_persona_seed()

        # create prompt
        creation_prompt_file = 'data/prompts.txt'
        with open(creation_prompt_file, 'r') as file:
            creation_prompt = file.read()
        
        # add persona template and seed
        creation_prompt += content + '\n\nSEED: ' + seed

        # get response from LLM
        # TODO: make model not hard-coded (get from config.ini)
        response = get_llm_response(
            config=self.config, 
            prompt=creation_prompt, 
            model='gpt-4o', 
            client=self.client)  

        data = json.loads(response)

        # TODO: there must be a cleaner way to dump a JSON into SQL

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

        # execute query
        res = self._execute_sql_query(
            query, 
            values=(
                first_name, last_name, backstory, hook,
                profile_pic_description, demographics, psychographics,
                personality, summary, interests, family
            ),
            fetch='one')
        self.uuid = res[0]

        self._log(f"INFO: CREATED AGENT {self.uuid}")
        
        # TODO: this won't work with current implementation
        # self.db_results = get_agent_row(uuid=self.uuid,cursor=cursor)


    def load_agent_details(self):
        """
        Retrieves and loads the details of an agent based on its UUID from the database.

        This function queries the 'agents' table in the DB using the agent's UUID
        then stores in self.db_results as a `dict`.

        Note:
        Before calling this method:
        - ensure 'uuid' is set AND VALID,
        - ensure the database connection (self.connection_pool) is set.
        """
        start_time = time.time()

        # Calculate and display elapsed time
        self._log("INFO: ====STARTING AGENT LOADING====")
        self._log(f"INFO: UUID {self.uuid}")

        # Query DB for agent details
        query = sql.SQL("SELECT * FROM agents WHERE uuid = %s LIMIT 1;")
        res = self._execute_sql_query(query, values=(self.uuid,), 
                                      fetch='all', cf=RealDictCursor)

        # Get results from the cursor object (will be a list of dicts)
        # and store 0-th dictionary
        self.db_results = res[0]

        # TODO: wouldn't it be better to just do cursor.fetchone()?

        # calculate sub-elapsed time and log.
        sub_time = time.time()
        elapsed_time = sub_time - start_time
        self._log(f"INFO: complete agent details load. UUID: {self.uuid}. \
                  Elapsed Time: {elapsed_time:.5g}")

        # Start loading memories
        self._log(f"INFO: start load agent memories.")

        # Load Memories for Agent
        # Note: This function was redundant and thus removed: MGY: 20240724
        #self.load_faiss_memories_object()  
        self.load_memory_vector_db()

        # calculate sub-elapsed time and log.
        sub_time2 = time.time()
        elapsed_time2 = sub_time2 - sub_time
        self._log(f"INFO: complete load agent memories. \
                  Elapsed Time: {elapsed_time2:.5g} ")

        # Log events
        self._log("INFO: ======AGENT LOADING COMPLETE====")


    def nullify_current_event(self):
        pass


    def load_memory_vector_db(self):
        """
        Loads a dictionary of memory descriptions into a FAISS index 
        for efficient similarity search.
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