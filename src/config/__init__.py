import yaml
import os
import sys
from typing import Dict, Any, Optional
# Get the absolute path to the directory one level above the current file's directory
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(MAIN_DIR)

# Importing logging utilities (assuming they are implemented in 'utils')
from utils import pipeline_log, error_log

def config_yaml_reader(file_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Reads a YAML configuration file and returns its contents as a dictionary.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        Dict[str, Any]: Parsed content of the YAML file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        yaml.YAMLError: If there is an error while parsing the YAML file.
    """
    try:
        # If not, look for any YAML file in the specified directory
        config_files = os.listdir("/workspaces/Chatbot-Restaurant/config")
        for con in config_files:
            if con.endswith(".yaml"):
                # Set the first YAML file found as the configuration file
                file_path = os.path.join("/workspaces/Chatbot-Restaurant/config", con)
                pipeline_log.info(f"Using alternative configuration file: {file_path}")
                break
            else:
                # If no YAML file is found, raise an error
                error_msg = "No YAML configuration file found in the directory."
                error_log.error(error_msg)
                raise FileNotFoundError(error_msg)

        # Load the YAML file
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
            pipeline_log.info("Successfully loaded configuration from YAML file.")
            return config
    except FileNotFoundError as e:
            error_log.error(f"File not found error: {e}")
            raise
    except yaml.YAMLError as e:
        error_log.error(f"Error parsing YAML file: {e}")
        raise ValueError(f"Error parsing YAML file: {e}")

class Config:
    """
    Class to handle configuration loading and distribution into sections.
    """
    def __init__(self, file_path: Optional[str] = None):
        """
        Initialize the configuration loader.

        Args:
            file_path (Optional[str]): Path to the YAML configuration file. 
                                   If not provided, searches for a YAML file in the default directory.
        Raises:
            RuntimeError: If configuration loading fails.
        """
        try:
            self.config = config_yaml_reader(file_path)
            pipeline_log.info("Configuration loaded successfully.")
        except Exception as e:
            error_log.error(f"Failed to initialize configuration: {e}")
            raise RuntimeError(f"Failed to initialize configuration: {e}")

        # First Distribution of the Configuration Dictionary
        self.project = self.config.get("project", {})
        self.database = self.config.get("database", {})
        self.embedding_model = self.config.get("embedding_model", {})
        self.retrieval = self.config.get("retrieval", {})
        self.generative_model = self.config.get("generative_model", {})
        self.api = self.config.get("api", {})
        
        # Project
        self.name = self.project["name"]
        self.version = self.project["version"]
        self.description = self.project["description"]
        pipeline_log.info("Project configuration loaded successfully.")

        # Database
        self.type = self.database["type"]
        self.path_database = self.database["path"]
        self.FAQsQ = self.database["FAQsQ"]
        self.columns_faqs = self.database["columns_faqs"]
        self.MENUITEMsQ = self.database["MENUITEMsQ"]
        self.columns_menuitems = self.database["columns_menuitems"]
        pipeline_log.info("Database configuration loaded successfully.")

        # Retrieval 
        self.top_k_results = self.retrieval["top_k_results"]
        pipeline_log.info("Retrieval configuration loaded successfully.")

        # Generative Model
        self.max_length = self.generative_model["max_length"]
        self.do_sample = self.generative_model["do_sample"]
        self.temperature = self.generative_model["temperature"]
        self.top_k = self.generative_model["top_k"]
        self.top_p = self.generative_model["top_p"]
        pipeline_log.info("Generative model configuration loaded successfully.")

        # API
        self.host = self.api["host"]
        self.port = self.api["port"]
        pipeline_log.info("API configuration loaded successfully.")


if __name__ == "__main__":
    configration = Config(file_path=None)
    print(f"Host API :{configration.host}")
