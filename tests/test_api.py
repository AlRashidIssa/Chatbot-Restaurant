import sys
import pytest
from fastapi.testclient import TestClient
# Add your project directory to sys.path
sys.path.append("/workspaces/Chatbot-Restaurant/")
from src.api.app import app  

# Create a TestClient instance
client = TestClient(app)

# Test root route (GET request)
def test_get_form():
    response = client.get("/")
    assert response.status_code == 200
    assert "form" in response.text  # Assuming your template renders a form

# Test /chat route (POST request)
def test_chat():
    query = "What is the menu?"
    response = client.post("/chat", data={"query": query})
    
    assert response.status_code == 200
    assert "response" in response.text  # Assuming 'response' is part of the response template
