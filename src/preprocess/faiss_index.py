"""
Create FAISS Index
"""

import os
import sys
import faiss
import numpy as np
from abc import ABC, abstractmethod

# Get the absolute path to the directory one level above the current file's directory
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(MAIN_DIR)

# Importing logging utilities (assuming they are implemented in 'utils')
from utils import pipeline_log, error_log


class IFAISSIndex(ABC):
    """
    Abstract interface for creating and managing a FAISS index.
    """

    @abstractmethod
    def create_faiss_index(self, embedding_array: np.ndarray) -> faiss.IndexFlatL2:
        """
        Abstract method to create a FAISS index for fast similarity search.

        Args:
            embedding_array (np.ndarray): NumPy array containing embeddings.

        Returns:
            faiss.IndexFlatL2: A FAISS index object for similarity search.

        Raises:
            ValueError: If the input embeddings are not valid.
        """
        pass


class FAISSIndex(IFAISSIndex):
    """
    Implementation of FAISS index creation and management.
    """

    def create_faiss_index(self, embedding_array: np.ndarray) -> faiss.IndexFlatL2:
        """
        Creates a FAISS index for fast similarity search using L2 distance.

        Args:
            embedding_array (np.ndarray): NumPy array containing embeddings. 
                Shape should be (num_samples, embedding_dimension).

        Returns:
            faiss.IndexFlatL2: A FAISS index object for similarity search.

        Raises:
            ValueError: If the input embeddings are not valid.
            RuntimeError: If an unexpected error occurs during index creation.
        """
        # Input Validation
        if not isinstance(embedding_array, np.ndarray):
            error_msg = "The 'embedding_array' must be a NumPy array."
            error_log.error(error_msg)
            raise ValueError(error_msg)

        if len(embedding_array.shape) != 2:
            error_msg = "The 'embedding_array' must be a 2D NumPy array with shape (num_samples, embedding_dimension)."
            error_log.error(error_msg)
            raise ValueError(error_msg)

        # Log the start of FAISS index creation
        pipeline_log.info(
            f"Starting FAISS index creation. Embedding array shape: {embedding_array.shape}"
        )

        try:
            # Create FAISS index
            embedding_dimension = embedding_array.shape[1]
            index = faiss.IndexFlatL2(embedding_dimension)
            pipeline_log.info("FAISS index created successfully.")

            # Add embeddings to the index
            index.add(embedding_array)
            pipeline_log.info("Embeddings added to the FAISS index successfully.")

        except Exception as e:
            error_msg = f"An error occurred while creating the FAISS index: {e}"
            error_log.error(error_msg)
            raise RuntimeError(error_msg)

        # Log the successful creation of the FAISS index
        pipeline_log.info("FAISS index creation completed successfully.")

        # Return the FAISS index
        return index


# Example Usage
if __name__ == "__main__":
    # Example embeddings
    num_samples = 100
    embedding_dimension = 384
    example_embeddings = np.random.rand(num_samples, embedding_dimension).astype("float32")

    # Initialize the FAISSIndex class
    faiss_index_creator = FAISSIndex()

    # Create FAISS index
    try:
        index = faiss_index_creator.create_faiss_index(example_embeddings)
        print(f"FAISS index created. Number of vectors: {index.ntotal}")
    except Exception as e:
        print(f"An error occurred: {e}")