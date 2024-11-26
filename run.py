import subprocess
import time
from multiprocessing import Process

def run_log_api():
    # Run FastAPI api_logs in a separate process using Uvicorn
    subprocess.run(["uvicorn", "monitoring.api_logs:app", "--reload", "--port", "8080"])

def run_chatbot_api():
    # Run FastAPI chatbot in a separate process using Uvicorn
    subprocess.run(["uvicorn", "src.api.app:app", "--reload", "--port", "5000"])

if __name__ == "__main__":
    # Run APIs concurrently using separate processes
    log_api_process = Process(target=run_log_api)
    chatbot_api_process = Process(target=run_chatbot_api)

    log_api_process.start()
    chatbot_api_process.start()

    # Optionally, wait for both processes to finish
    log_api_process.join()
    chatbot_api_process.join()
