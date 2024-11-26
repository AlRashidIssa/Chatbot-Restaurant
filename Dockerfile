# Use Arch Linux as the base image
FROM archlinux:latest

# Set the working directory in the container
WORKDIR /workspaces/Chatbot-Restaurant

# Update the package database and install required dependencies
RUN pacman -Syu --noconfirm && \
    pacman -S --noconfirm base-devel git python python-pip python-virtualenv

# Copy the current directory into the container
COPY . /workspaces/Chatbot-Restaurant

# Set up Python virtual environment and install dependencies
RUN python -m venv EnV && \
    /workspaces/Chatbot-Restaurant/EnV/bin/pip install --no-cache-dir -r requirements.txt

# Run Tests
RUN /workspaces/Chatbot-Restaurant/EnV/bin/pytest -v tests/

# Expose the application port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "src.api.app:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
