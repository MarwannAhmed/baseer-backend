from pathlib import Path
import logging
import os

from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from fastapi import FastAPI

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

logger = logging.getLogger(__name__)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)

app = FastAPI(title="Baseer Backend")

logger.info("Application startup initiated")

logger.info("Downloading models from Azure Blob Storage...")

connection_string = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
blob_service_client = BlobServiceClient.from_connection_string(
    connection_string
)
logger.info("Connected to Azure Blob Storage")

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

    logger.info("Processing model: %s", model)

    for blob in container_client.list_blobs():
        file_name = blob.name
        logger.info("Processing file: %s", file_name)

        local_file = model_dir / file_name

        if local_file.exists():
            logger.info(
                "File %s already exists, skipping download.",
                local_file,
            )
            continue

        logger.info(
            "Downloading %s from Azure Blob Storage...",
            file_name,
        )

        blob_client = blob_service_client.get_blob_client(
            container=model,
            blob=file_name,
        )

        with open(local_file, "wb") as f:
            f.write(blob_client.download_blob().readall())

        logger.info(
            "Downloaded %s to %s",
            file_name,
            local_file,
        )
    

from app.routers import analyze
app.include_router(analyze.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
