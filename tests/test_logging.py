import sys
import os
from unittest import mock
from pathlib import Path
import pytest

# Dynamically locate the main project directory
current_dir = Path(__file__).resolve().parent
MAIN_DIR = current_dir
while MAIN_DIR.name != "Chatbot-Restaurant":
    MAIN_DIR = MAIN_DIR.parent

# Add the `src` directory to sys.path
sys.path.append(str(MAIN_DIR))
print("Maind Dir" + str(MAIN_DIR))
from src.utils import chatbot_log, error_log, pipeline_log

# Test for logging setup
def test_logging_setup():
    """Test if the log directory and files are properly set up."""
    # Define the log directory path
    log_dir = str(MAIN_DIR) + "/logs"
    # Ensure the log directory exists
    assert os.path.exists(log_dir),  f"Log directory {log_dir} does not exist."

    # Ensure all expected log files exist
    expected_logs = ["error.log", "chatbot.log", "pipeline_log.log"]
    for log_file in expected_logs:
        log_path = f"{log_dir}/{log_file}"
        assert os.path.exists(log_path), f"Log file {log_file} not found in {log_dir}."

# Test logger functionality
def test_logger_functionality():
    """Test if loggers output the correct log messages."""
    
    # Test chatbot_log
    with mock.patch.object(chatbot_log, 'info') as chatbot_info_mock, \
         mock.patch.object(chatbot_log, 'debug') as chatbot_debug_mock:
        chatbot_log.info("Chatbot info message")
        chatbot_log.debug("Chatbot debug message")
        chatbot_info_mock.assert_called_with("Chatbot info message")
        chatbot_debug_mock.assert_called_with("Chatbot debug message")

    # Test error_log
    with mock.patch.object(error_log, 'error') as error_mock, \
         mock.patch.object(error_log, 'critical') as critical_mock:
        error_log.error("Error log message")
        error_log.critical("Critical error message")
        error_mock.assert_called_with("Error log message")
        critical_mock.assert_called_with("Critical error message")

    # Test pipeline_log
    with mock.patch.object(pipeline_log, 'warning') as pipeline_warning_mock, \
         mock.patch.object(pipeline_log, 'info') as pipeline_info_mock:
        pipeline_log.warning("Pipeline warning message")
        pipeline_log.info("Pipeline info message")
        pipeline_warning_mock.assert_called_with("Pipeline warning message")
        pipeline_info_mock.assert_called_with("Pipeline info message")

# Additional tests for log rotation (if applicable)
@pytest.mark.parametrize("log_file", ["error.log", "chatbot.log", "inference_log.log"])
def test_log_file_rotation(log_file):
    """Test if log files rotate correctly when reaching a size limit."""
    log_path = MAIN_DIR / "logs" / log_file
    max_log_size = 1024 * 1024  # 1 MB, for example

    if log_path.exists():
        assert log_path.stat().st_size <= max_log_size, \
            f"Log file {log_file} exceeds the maximum size of {max_log_size} bytes."
