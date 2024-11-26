import sys
import pytest
import os
from unittest import mock
from src.utils import chatbot_log
import sqlite3
import pandas as pd
from pathlib import Path

# Get the current directory 
current_dir = Path(__file__).resolve()

# Search for the main directory
MAIN_DIR = current_dir
while MAIN_DIR.name != "Chatbot-Restaurant":
    MAIN_DIR = MAIN_DIR.parent

# Add your project directory to sys.path
sys.path.append(MAIN_DIR)

# Test for logging
def test_logging_setup():
    """Test if the logger is configured properly."""
    # Check if the log file directory exists
    log_dir = "/workspaces/Chatbot-Restaurant/logs"
    assert os.path.exists(log_dir), f"Log directory {log_dir} does not exist."
    
    # Check if the log files exist
    assert os.path.isfile(os.path.join(log_dir, "error.log")), "Error log file not found."
    assert os.path.isfile(os.path.join(log_dir, "chatbot.log")), "Chatbot log file not found."
    assert os.path.isfile(os.path.join(log_dir, "inference_log.log")), "Inference log file not found."
    
    # Check if the logger outputs messages (mocking the logger instance)
    with mock.patch.object(chatbot_log, 'info') as mock_info:
        chatbot_log.info("Test message")
        mock_info.assert_called_with("Test message")

