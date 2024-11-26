import logging
import os

from pathlib import Path

# Get the current directory 
current_dir = Path(__file__).resolve()

# Search for the main directory
MAIN_DIR = current_dir
while MAIN_DIR.name != "Chatbot-Restaurant":
    MAIN_DIR = MAIN_DIR.parent

# Define the log Directory relative to this file's location
LOG_DIR = f"{MAIN_DIR}/logs"
LOG_DIR = os.path.abspath(LOG_DIR)

os.makedirs(LOG_DIR, exist_ok=True)

def setup_logger(name, log_file, level=logging.INFO):
    """Sets up a logger with specified name, log file, and logging level, ensuring no duplicate handlers."""
    logger = logging.getLogger(name)
    
    # Check if the logger already has handlers
    if not logger.hasHandlers():
        logger.setLevel(level)

        # Create a file handler for logging
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)

        # Create a console handler for logging 
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # Define log message format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger

# Create loggers
error_log = setup_logger(name="error_log", log_file=os.path.join(LOG_DIR, "error.log"), level=logging.ERROR)
chatbot_log = setup_logger(name="chatbot_log", log_file=os.path.join(LOG_DIR, "chatbot.log"), level=logging.INFO)
pipeline_log = setup_logger(name="inference_log", log_file=os.path.join(LOG_DIR, "pipeline_log.log"), level=logging.INFO)
