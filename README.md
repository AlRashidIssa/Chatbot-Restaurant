# Restaurant Chatbot with RAG, FLAN-T5, and FFISSIndex

This project implements a chatbot system for restaurants in Saudi Arabia (KSA) using a **Retrieval-Augmented Generation (RAG)** approach. The system is powered by the **FLAN-T5** model for natural language understanding and **FFISSIndex** for efficient embedding-based search.

The project also includes a full pipeline for logging, error handling, and real-time monitoring of logs. Two APIs are created: one for handling chatbot requests and another for monitoring logs in real-time.

---

## Features

- **Chatbot with RAG**: Leverages FLAN-T5 and FFISSIndex to retrieve relevant data and generate human-like responses.
- **Full Pipeline**: Includes data retrieval, processing, error handling, and logging mechanisms.
- **Real-time Log Monitoring**: An API for monitoring real-time logs to track chatbot performance and errors.
- **Dockerized**: Fully containerized using Docker for streamlined deployment.
- **Unit Tests**: Reliability ensured with tests written using `pytest`.
- **Configuration with YAML**: Easily configurable via YAML files for flexibility.

---

## Requirements

Ensure you have the following dependencies installed before starting:

- Python 3.x
- Docker
- `pip` (Python package manager)

Install the necessary Python libraries using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

For real-time monitoring and logging, ensure you have access to your desired logging infrastructure (e.g., ELK Stack, Datadog, etc.).

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/restaurant-chatbot.git
cd restaurant-chatbot
```

2. Build the Docker image:

```bash
docker build -t restaurant-chatbot .
```

3. Run the `run.py` script to start both the Chatbot API and the Log Monitoring API concurrently:

```bash
python run.py
```

This will expose two APIs:

- **Chatbot API**: Accessible at `http://127.0.0.1:5000`
- **Real-time Log Monitoring API**: Accessible at `http://127.0.0.1:8080`

---

## API Endpoints

### Chatbot API (HTML Template Response)

- **Endpoint**: `/chat`
- **Method**: `GET`
- **Description**: Allows users to input a query, and the chatbot generates a response, returning the result as an HTML page.

### Real-time Log Monitoring API (HTML Template Response)

- **Endpoint**: `/logs`
- **Method**: `GET`
- **Description**: Retrieves the real-time logs of chatbot interactions, displayed in an HTML format.

---

## Testing

Unit tests are included in the project to ensure the functionality of the system. Run the tests with:

```bash
pytest
```

---

## Configuration

Configuration settings are managed through a `config.yaml` file, allowing easy modification of various settings, such as the model type, logging level, and embedding configurations.

Example settings include:

- Model configuration for FLAN-T5
- Logging settings
- Embedding configurations for FFISSIndex

---

## Contribution

Feel free to fork the repository and submit pull requests. When contributing, please adhere to the coding style and add tests for any new features or bug fixes.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## Acknowledgements

- **FLAN-T5**: A powerful language model for NLP tasks.
- **FFISSIndex**: A fast indexing library for efficient embedding-based search.
- **Docker**: For containerization and easy deployment.
- **pytest**: For writing and running unit tests.