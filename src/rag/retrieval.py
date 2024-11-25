"""
Retrieve Results from FAISS Indices.
"""

import os
import sys
import pandas as pd
import numpy as np
import faiss
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from sentence_transformers import SentenceTransformer

# Get the absolute path to the directory one level above the current file's directory
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(MAIN_DIR)

# Importing logging utilities (assuming they are implemented in 'utils')
from utils import pipeline_log, error_log


class IRetrieve(ABC):
    """
    Abstract interface for retrieving results from FAISS indices based on a query.
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        embedding_model: SentenceTransformer,
        index_1: faiss.IndexFlatL2,
        index_2: faiss.IndexFlatL2,
        df_index_1: pd.DataFrame,
        df_index_2: pd.DataFrame,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Abstract method to retrieve results from two FAISS indices.

        Args:
            query (str): The search query.
            embedding_model (SentenceTransformer): Pretrained embedding model.
            index_1 (faiss.IndexFlatL2): First FAISS index for retrieval.
            index_2 (faiss.IndexFlatL2): Second FAISS index for retrieval.
            df_index_1 (pd.DataFrame): DataFrame corresponding to the first FAISS index.
            df_index_2 (pd.DataFrame): DataFrame corresponding to the second FAISS index.
            top_k (int): Number of top results to retrieve from each index. Default is 3.

        Returns:
            Dict[str, Any]: A dictionary containing the query and retrieved results.

        Raises:
            ValueError: If inputs are not valid.
        """
        pass


class Retrieve(IRetrieve):
    """
    Implementation for retrieving results from FAISS indices based on a query.
    """

    def retrieve(
        self,
        query: str,
        embedding_model: SentenceTransformer,
        index_1: faiss.IndexFlatL2,
        index_2: faiss.IndexFlatL2,
        df_index_1: pd.DataFrame,
        df_index_2: pd.DataFrame,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Retrieves results from two FAISS indices based on the query.

        Args:
            query (str): The search query.
            embedding_model (SentenceTransformer): Pretrained embedding model.
            index_1 (faiss.IndexFlatL2): First FAISS index for retrieval.
            index_2 (faiss.IndexFlatL2): Second FAISS index for retrieval.
            df_index_1 (pd.DataFrame): DataFrame corresponding to the first FAISS index.
            df_index_2 (pd.DataFrame): DataFrame corresponding to the second FAISS index.
            top_k (int): Number of top results to retrieve from each index. Default is 3.

        Returns:
            Dict[str, Any]: A dictionary containing the query and retrieved results.

        Raises:
            ValueError: If inputs are not valid or indices encounter errors.
        """
        # Validate input types
        if not isinstance(query, str):
            error_msg = "The 'query' must be a string."
            error_log.error(error_msg)
            raise ValueError(error_msg)

        if not isinstance(embedding_model, SentenceTransformer):
            error_msg = "The 'embedding_model' must be an instance of SentenceTransformer."
            error_log.error(error_msg)
            raise ValueError(error_msg)

        if not isinstance(index_1, faiss.IndexFlatL2) or not isinstance(index_2, faiss.IndexFlatL2):
            error_msg = "The 'index_1' and 'index_2' must be instances of faiss.IndexFlatL2."
            error_log.error(error_msg)
            raise ValueError(error_msg)

        if not isinstance(df_index_1, pd.DataFrame) or not isinstance(df_index_2, pd.DataFrame):
            error_msg = "The 'df_index_1' and 'df_index_2' must be pandas DataFrames."
            error_log.error(error_msg)
            raise ValueError(error_msg)

        if not isinstance(top_k, int) or top_k <= 0:
            error_msg = "The 'top_k' must be a positive integer."
            error_log.error(error_msg)
            raise ValueError(error_msg)

        pipeline_log.info("Starting retrieval process.")

        try:
            # Encode the query into embeddings
            query_embedding = embedding_model.encode(query).reshape(1, -1)
            pipeline_log.info(f"Query encoded successfully: {query}")

            # Retrieve from index_1
            index_1_distances, index_1_indices = index_1.search(query_embedding, top_k)
            index_1_results = df_index_1.iloc[index_1_indices[0]].to_dict(orient="records")
            pipeline_log.info(f"Top {top_k} results retrieved from index_1.")

            # Retrieve from index_2
            index_2_distances, index_2_indices = index_2.search(query_embedding, top_k)
            index_2_results = df_index_2.iloc[index_2_indices[0]].to_dict(orient="records")
            pipeline_log.info(f"Top {top_k} results retrieved from index_2.")

            # Combine results into a dictionary
            results = {
                "query": query,
                "_1_result": index_1_results,
                "_2_result": index_2_results,
            }

            pipeline_log.info("Retrieval process completed successfully.")
            return results

        except Exception as e:
            error_msg = f"An error occurred during the retrieval process: {e}"
            error_log.error(error_msg)
            raise RuntimeError(error_msg)



# Test Case Just for Test Method and Class Operation.
if __name__ == "__main__":
    # Example embeddings and DataFrames
    num_samples = 100
    embedding_dimension = 384
    example_embeddings_1 = np.random.rand(num_samples, embedding_dimension).astype("float32")
    example_embeddings_2 = np.random.rand(num_samples, embedding_dimension).astype("float32")
    
    example_df_1 = pd.DataFrame({"data": [f"Sample {i}" for i in range(num_samples)]})
    example_df_2 = pd.DataFrame({"data": [f"Example {i}" for i in range(num_samples)]})

    # Create FAISS indices
    index_1 = faiss.IndexFlatL2(embedding_dimension)
    index_1.add(example_embeddings_1)

    index_2 = faiss.IndexFlatL2(embedding_dimension)
    index_2.add(example_embeddings_2)

    # Load a SentenceTransformer model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Initialize the Retrieve class
    retriever = Retrieve()

    # Perform retrieval
    try:
        results = retriever.retrieve(
            query="Find similar items",
            embedding_model=embedding_model,
            index_1=index_1,
            index_2=index_2,
            df_index_1=example_df_1,
            df_index_2=example_df_2,
            top_k=5,
        )
        print("Results:", results)
    except Exception as e:
        print("An error occurred:", e)