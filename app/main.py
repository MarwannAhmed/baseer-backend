from pathlib import Path
import os

from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from fastapi import FastAPI

load_dotenv()

app = FastAPI(title="Baseer Backend")

def download_models() -> None:
    print("Downloading models from Azure Blob Storage...")

    connection_string = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    blob_service_client = BlobServiceClient.from_connection_string(
        connection_string
    )
    print("Connected to Azure Blob Storage")

    models = [
        "svm-text-en",
        "cnn-ocr",
        "object",
        "ar-pp-ocrv5-mobile-rec-infer",
        "langmodel-ocr",
        "pp-ocrv5-mobile-det-infer"
    ]
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    for model in models:
        container_client = blob_service_client.get_container_client(model)
        if model == "langmodel-ocr":
            model_dir = models_dir / "ocr" / "classical" / "langmodel"
        else:
            model_dir = models_dir / model
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"Processing model: {model}")

        for blob in container_client.list_blobs():
            file_name = blob.name
            print(f" - Processing file: {file_name}")
            local_file = model_dir / file_name
            if local_file.exists():
                print(f"    File {local_file} already exists, skipping download.")
                continue

            print(f"    Downloading {file_name} from Azure Blob Storage...")
            blob_client = blob_service_client.get_blob_client(
                container=model,
                blob=file_name,
            )
            with open(local_file, "wb") as f:
                f.write(blob_client.download_blob().readall())
            print(f"    Downloaded {file_name} to {local_file}")

@app.on_event("startup")
def startup() -> None:
    download_models()

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
