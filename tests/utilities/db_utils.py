import psycopg2.pool    # get_connection_pool

def get_connection_pool(config):
    """
    Creates and returns a database connection pool using psycopg2 based on provided configuration details.

    Parameters:
    - config (ConfigParser): An object containing database configuration details. 
    Expected to have 'DB' section with 'DB_NAME', 'DB_USER', and 'DB_PASS' keys.

    Returns:
    - psycopg2.pool.SimpleConnectionPool: A connection pool object with a 
    minimum of 1 connection and a maximum of 25 connections.

    Example usage:
    config = configparser.ConfigParser()
    config.read('config.ini')
    pool = get_connection_pool(config)
    """
    # Create a DB connection pool based on config details
    connection_pool = psycopg2.pool.SimpleConnectionPool(
        1,  # minconn
        25,  # maxconn
        host=config.get('DB', 'DB_HOST'),
        port=config.get('DB', 'DB_PORT'),
        dbname=config.get('DB', 'DB_NAME'),
        user=config.get('DB', 'DB_USER'),
        password=config.get('DB', 'DB_PASS')
    )

    return connection_pool