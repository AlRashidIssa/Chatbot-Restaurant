import os
import sys
from transformers import pipeline, AutoModelForSeq2SeqLM
from abc import ABC, abstractmethod

# Get the absolute path to the directory one level above the current file's directory
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(MAIN_DIR)

# Importing logging utilities (assuming they are implemented in 'utils')
from utils import pipeline_log, error_log


class IFlanT5Load(ABC):
    """
    Abstract base class for loading a FLAN-T5 model.
    Defines an abstract method for model loading which needs to be implemented.
    """

    @abstractmethod
    def load(self) -> AutoModelForSeq2SeqLM:
        """
        Abstract method to load the FLAN-T5 model.

        Returns:
            AutoModelForSeq2SeqLM: The loaded transformer model.
        """
        pass

class FlanT5Load(IFlanT5Load):
    """
    Concrete class for loading a FLAN-T5 model for text-to-text generation tasks.
    Implements the 'load' method to load the model from Hugging Face's Model Hub.
    """

    def load(self) -> AutoModelForSeq2SeqLM:
        """
        Loads the FLAN-T5 model and prepares the text-to-text pipeline for inference.

        Returns:
            pipeline: A Hugging Face pipeline for text-to-text generation using FLAN-T5.

        Raises:
            Exception: If the model fails to load or an error occurs during initialization.
        """
        try:
            # Attempt to load the FLAN-T5 model from Hugging Face Model Hub
            pipeline_log.info("Loading FLAN-T5 model...")
            model_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

            # Log success
            pipeline_log.info("Successfully loaded the FLAN-T5 model.")
            return model_pipeline
        except Exception as e:
            # Log the error in case of failure
            error_log.error(f"Failed to load FLAN-T5 model: {str(e)}")
            error_log.error(f"Error loading FLAN-T5 model: {str(e)}")
            raise Exception(f"Error loading FLAN-T5 model: {str(e)}") from e

if __name__ == "__main__":
    model = FlanT5Load().load()
    model