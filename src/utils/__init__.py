import logging
import os

# Define the log Directory relative to this file's location
LOG_DIR = os.path.join(os.path.dirname(__file__), "/workspaces/Chatbot-Restaurant/logs")
LOG_DIR = os.path.abspath(LOG_DIR)
print("Log Directory absolute path:", LOG_DIR)

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
pipeline_log = setup_logger(name="inference_log", log_file=os.path.join(LOG_DIR, "inference_log.log"), level=logging.INFO)
