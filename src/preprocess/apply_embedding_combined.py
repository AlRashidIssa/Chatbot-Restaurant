"""
Apply Embedding for Combined DataFrame.
"""

import os
import sys
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer

# Get the absolute path to the directory one level above the current file's directory
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(MAIN_DIR)

# Importing logging utilities (assuming they are implemented in 'utils')
from utils import pipeline_log, error_log


class IEmbeddingForCombined(ABC):
    """
    Abstract interface for applying embeddings to a specified column in a DataFrame.
    """

    @abstractmethod
    def embedded(
        self, embedding_model: SentenceTransformer, df: pd.DataFrame, column: str
    ) -> np.ndarray:
        """
        Abstract method to apply embeddings to a specified column.

        Args:
            embedding_model (SentenceTransformer): Pretrained SentenceTransformer model for generating embeddings.
            df (pd.DataFrame): Input DataFrame containing the data.
            column (str): Name of the column to apply embeddings to.

        Returns:
            np.ndarray: NumPy array with array embeddings.

        Raises:
            ValueError: If inputs are not of the expected type or column does not exist in the DataFrame.
        """
        pass


class EmbeddingForCombined(IEmbeddingForCombined):
    """
    Implementation of embedding application for a specified DataFrame column.
    """

    def embedded(
        self, embedding_model: SentenceTransformer, df: pd.DataFrame, column: str = "combined"
    ) -> np.ndarray:
        """
        Applies embeddings to a specified column in the DataFrame.

        Args:
            embedding_model (SentenceTransformer): Pretrained SentenceTransformer model for generating embeddings.
            df (pd.DataFrame): Input DataFrame containing the data.
            column (str): Name of the column to apply embeddings to.

        Returns:
            np.ndarray: NumPy array with array embeddings.

        Raises:
            ValueError: If inputs are not valid or if column does not exist in the DataFrame.
        """
        # Input Validation
        if not isinstance(embedding_model, SentenceTransformer):
            error_msg = "The 'embedding_model' must be an instance of SentenceTransformer."
            error_log.error(error_msg)
            raise ValueError(error_msg)

        if not isinstance(df, pd.DataFrame):
            error_msg = "The 'df' must be a pandas DataFrame."
            error_log.error(error_msg)
            raise ValueError(error_msg)

        if not isinstance(column, str):
            error_msg = "The 'column' must be a string representing a column name."
            error_log.error(error_msg)
            raise ValueError(error_msg)

        if column not in df.columns:
            error_msg = f"The column '{column}' does not exist in the DataFrame."
            error_log.error(error_msg)
            raise ValueError(error_msg)

        # Log the start of the embedding process
        pipeline_log.info(f"Starting embedding for column: {column}")

        try:
            # Apply embeddings to the specified column
            df[f"{column}_embedding"] = df[column].apply(
                lambda x: embedding_model.encode(x)
            )
            pipeline_log.info(f"Successfully generated embeddings for column: {column}")

            # Collect all embeddings into a list
            embeddings_list = df[f"{column}_embedding"].tolist()

            # Stack the embeddings into a NumPy array
            embeddings_array = np.vstack(embeddings_list)
            pipeline_log.info("Successfully converted embeddings to a NumPy array.")

        except Exception as e:
            error_msg = f"An error occurred while generating embeddings: {e}"
            error_log.error(error_msg)
            raise RuntimeError(error_msg)

        # Return the NumPy array of embeddings
        return embeddings_array


# Example usage
if __name__ == "__main__":
    # Example data
    data = {
        "combined": [
            "Alice 25 New York",
            "Bob 30 Los Angeles",
            "Charlie 35 Chicago",
        ]
    }
    df = pd.DataFrame(data)

    # Load a pre-trained SentenceTransformer model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Initialize the EmbeddingForCombined class
    embedder = EmbeddingForCombined()

    # Apply embeddings to the 'combined' column
    result = embedder.embedded(embedding_model, df, "combined")

    # Print the resulting NumPy array shape
    print(result.shape)
