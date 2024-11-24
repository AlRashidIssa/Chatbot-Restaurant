"""
Apply Embedding for Combined DataFrame.
"""

import os
import sys
import pandas as pd
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
    ) -> pd.DataFrame:
        """
        Abstract method to apply embeddings to a specified column.

        Args:
            embedding_model (SentenceTransformer): Pretrained SentenceTransformer model for generating embeddings.
            df (pd.DataFrame): Input DataFrame containing the data.
            column (str): Name of the column to apply embeddings to.

        Returns:
            pd.DataFrame: DataFrame with the embeddings applied to the specified column.

        Raises:
            ValueError: If inputs are not of the expected type or column does not exist in the DataFrame.
        """
        pass


class EmbeddingForCombined(IEmbeddingForCombined):
    """
    Implementation of embedding application for a specified DataFrame column.
    """

    def embedded(
        self, embedding_model: SentenceTransformer, df: pd.DataFrame, column: str
    ) -> pd.DataFrame:
        """
        Applies embeddings to a specified column in the DataFrame.

        Args:
            embedding_model (SentenceTransformer): Pretrained SentenceTransformer model for generating embeddings.
            df (pd.DataFrame): Input DataFrame containing the data.
            column (str): Name of the column to apply embeddings to.

        Returns:
            pd.DataFrame: DataFrame with a new column '{column}_embedding' containing the embeddings.

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
        except Exception as e:
            error_msg = f"An error occurred while generating embeddings: {e}"
            error_log.error(error_msg)
            raise RuntimeError(error_msg)

        # Return the DataFrame with the new embeddings column
        return df


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
    result_df = embedder.embedded(embedding_model, df, "combined")

    # Print the resulting DataFrame
    print(result_df.head())