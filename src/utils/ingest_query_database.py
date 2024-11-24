import os
import sys
import sqlite3
import pandas as pd
from sentence_transformers import SentenceTransformer
from abc import ABC, abstractmethod

# Get the absolute path to the directory one level above the current file's directory
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(MAIN_DIR)

# Importing logging utilities (assuming they are implemented in 'utils')
from utils import pipeline_log, error_log

class IIngestQueryDatabase(ABC):
    """
    Abstract base class for ingesting data from a database using SQL queries.
    Defines an abstract method `ingest` that must be implemented by subclasses.
    """

    @abstractmethod
    def ingest(self, db_path: str, query: str) -> pd.DataFrame:
        """
        Abstract method to execute a SQL query on the provided database.

        Args:
            db_path (str): The path to the SQLite database file.
            query (str): The SQL query to execute.

        Returns:
            pd.DataFrame: The result of the SQL query as a pandas DataFrame.

        Raises:
            ValueError: If the `db_path` or `query` are not valid.
            FileNotFoundError: If the SQLite database file does not exist.
            sqlite3.Error: If there is a database-related error during query execution.
        """
        pass


class IngestQueryDatabase(IIngestQueryDatabase):
    """
    Concrete class for ingesting data from an SQLite database.
    Implements the `ingest` method to execute SQL queries and return results as DataFrames.
    """

    def ingest(self, db_path: str, query: str) -> pd.DataFrame:
        """
        Executes a SQL query on the provided SQLite database and returns the result as a pandas DataFrame.

        Args:
            db_path (str): The path to the SQLite database file.
            query (str): The SQL query to execute.

        Returns:
            pd.DataFrame: The result of the SQL query as a pandas DataFrame.

        Raises:
            ValueError: If the `db_path` or `query` are invalid.
            FileNotFoundError: If the SQLite database file does not exist.
            sqlite3.Error: If there is a database-related error during query execution.
        """
        # Input validation
        if not isinstance(db_path, str):
            error_msg = "The 'db_path' must be a string."
            error_log.error(error_msg)
            raise ValueError(error_msg)

        if not isinstance(query, str):
            error_msg = "The 'query' must be a string."
            error_log.error(error_msg)
            raise ValueError(error_msg)

        if not os.path.exists(db_path):
            error_msg = f"The database file at {db_path} does not exist."
            error_log.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            # Connect to the SQLite database
            pipeline_log.info(f"Connecting to the SQLite database at {db_path}...")
            conn = sqlite3.connect(db_path)

            # Execute the query and load the result into a pandas DataFrame
            pipeline_log.info(f"Executing query: {query}")
            table_df = pd.read_sql_query(query, conn)

            # Close the database connection
            conn.close()
            pipeline_log.info("Query executed successfully and database connection closed.")

            # Log the successful ingestion
            pipeline_log.info(f"Data ingested successfully from {db_path} with query: {query[:50]}...")  # Log first 50 chars of query for brevity
            return table_df

        except sqlite3.Error as e:
            # Log and raise database-related errors
            error_msg = f"SQLite error occurred: {str(e)}"
            error_log.error(error_msg)
            raise sqlite3.Error(error_msg) from e
        except Exception as e:
            # Log and raise any unexpected errors
            error_msg = f"An unexpected error occurred: {str(e)}"
            error_log.error(error_msg)
            raise Exception(error_msg) from e


# Example usage (assuming this is a script entry point)
if __name__ == "__main__":
    # Path to the SQLite database and SQL query
    db_path = "/workspaces/Chatbot-Restaurant/database/restaurant.db"
    query = "SELECT * FROM faqs;"

    # Initialize the IngestQueryDatabase class and ingest data
    ingest_query = IngestQueryDatabase()
    faqs_df = ingest_query.ingest(db_path, query)

    # Print the result
    print(f"Ingested data:\n{faqs_df.head()}")