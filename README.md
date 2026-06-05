## Requirements
- Python 3.11

## Setup
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Environment Variable
- Create a .env file:
```bash
AZURE_STORAGE_CONNECTION_STRING=your_connection_string
```

## Run (dev)
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
