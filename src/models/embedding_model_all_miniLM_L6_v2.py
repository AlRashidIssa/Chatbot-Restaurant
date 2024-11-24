import os
import sys
from sentence_transformers import SentenceTransformer
from abc import ABC, abstractmethod

# Get the absolute path to the directory one level above the current file's directory
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(MAIN_DIR)

# Importing logging utilities (assuming they are implemented in 'utils')
from utils import pipeline_log, error_log

class IEmbeddingLoader(ABC):
    """
    Abstract base class for loading sentence transformers for generating embeddings.
    """

    @abstractmethod
    def load(self) -> SentenceTransformer:
        """
        Abstract method to load a SentenceTransformer model.

        Returns:
            SentenceTransformer: The loaded SentenceTransformer model.
        """
        pass

class EmbeddingLoader(IEmbeddingLoader):
    """
    Concrete class to load the SentenceTransformer model `all-MiniLM-L6-v2`
    and handle generating sentence embeddings.
    """

    def load(self) -> SentenceTransformer:
        """
        Loads the SentenceTransformer model (`all-MiniLM-L6-v2`) and prepares it for generating sentence embeddings.

        Returns:
            SentenceTransformer: The loaded SentenceTransformer model.

        Raises:
            Exception: If the model fails to load or an error occurs during initialization.
        """
        try:
            # Attempt to load the SentenceTransformer model
            pipeline_log.info("Loading SentenceTransformer model: all-MiniLM-L6-v2...")
            model = SentenceTransformer("all-MiniLM-L6-v2")

            # Log success
            pipeline_log.info("Successfully loaded the SentenceTransformer model.")
            pipeline_log.info("SentenceTransformer model 'all-MiniLM-L6-v2' loaded successfully.")
            return model
        except Exception as e:
            # Log the error if loading fails
            error_log.error(f"Failed to load SentenceTransformer model: {str(e)}")
            raise Exception(f"Error loading SentenceTransformer model: {str(e)}") from e

if __name__ == "__main__":
    embedded = EmbeddingLoader().load()
    embedded