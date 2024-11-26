from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os

# Define log directory
log_dir = "/workspaces/Chatbot-Restaurant/logs"

# Initialize FastAPI app
app = FastAPI()

# Set up templates
templates = Jinja2Templates(directory="/workspaces/Chatbot-Restaurant/logs/templates")

# Function to read all logs
# Function to read all logs
def get_all_logs():
    logs = {}
    if os.path.exists(log_dir):
        for file_name in os.listdir(log_dir):
            if file_name.endswith(".log"):
                file_path = os.path.join(log_dir, file_name)
                with open(file_path, "r") as file:
                    logs[file_name] = file.read()
    # Sort logs with "error.log" on top
    sorted_logs = dict(sorted(logs.items(), key=lambda x: (x[0] != "error.log", x[0])))
    return sorted_logs



# Route to render the logs dashboard
@app.get("/", response_class=HTMLResponse)
async def display_logs(request: Request):
    logs = get_all_logs()  # Read logs dynamically
    return templates.TemplateResponse("index.html", {"request": request, "logs": logs})


# API to get logs in JSON format
@app.get("/api/logs")
async def api_logs():
    logs = get_all_logs()
    return logs
