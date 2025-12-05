import os
import time
import glob
import concurrent.futures
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from google import genai
from google.genai import types

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DOCUMENTS_DIR = "/Users/macbookpro16_stic_admin/Documents/file_search_latest/documents"
API_KEY_ENV_VAR = "GOOGLE_API_KEY"
MODEL_ID = "gemini-2.5-flash"

# Global state to hold the client and store name
app_state = {
    "client": None,
    "store_name": None,
    "ready": False
}

def get_api_key():
    """Retrieves the API key from the environment."""
    api_key = os.environ.get(API_KEY_ENV_VAR)
    if not api_key:
        logger.error(f"Error: {API_KEY_ENV_VAR} environment variable not set.")
        return None
    return api_key

def upload_and_monitor(client, file_path, store_name):
    """Uploads a single file to the store and monitors its status."""
    file_name = os.path.basename(file_path)
    logger.info(f"Starting upload for: {file_name}")
    
    try:
        # Strictly following docs: pass file path string and config as dict
        operation = client.file_search_stores.upload_to_file_search_store(
            file=file_path,
            file_search_store_name=store_name,
            config={'display_name': file_name}
        )
        
        logger.info(f"Upload initiated for {file_name}. Operation: {operation.name}")
        
        # Poll for completion
        while not operation.done:
            time.sleep(2)
            # Pass the operation object itself to get()
            operation = client.operations.get(operation)
            
        if operation.error:
            logger.error(f"Error processing {file_name}: {operation.error}")
            return None
        
        logger.info(f"Successfully processed: {file_name}")
        return file_name
        
    except Exception as e:
        logger.error(f"Exception uploading {file_name}: {e}")
        return None

def setup_rag_system():
    """Initializes the RAG system: creates store and uploads files."""
    api_key = get_api_key()
    if not api_key:
        logger.error("API Key missing. RAG system setup failed.")
        return

    client = genai.Client(api_key=api_key)
    app_state["client"] = client
    logger.info("Client initialized.")

    # 1. Create File Search Store
    logger.info("Creating File Search Store...")
    try:
        file_search_store = client.file_search_stores.create(
            config={'display_name': 'fastapi_knowledge_base'}
        )
        app_state["store_name"] = file_search_store.name
        logger.info(f"Created store: {file_search_store.name}")
    except Exception as e:
        logger.error(f"Failed to create store: {e}")
        return

    # 2. List files
    pdf_files = glob.glob(os.path.join(DOCUMENTS_DIR, "*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {DOCUMENTS_DIR}")
        app_state["ready"] = True # Still ready, just empty
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files.")

    # 3. Upload Files (Parallelized)
    successful_uploads = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_file = {
            executor.submit(upload_and_monitor, client, f, file_search_store.name): f 
            for f in pdf_files
        }
        
        for future in concurrent.futures.as_completed(future_to_file):
            result = future.result()
            if result:
                successful_uploads += 1

    logger.info(f"Uploads complete. {successful_uploads}/{len(pdf_files)} files processed successfully.")
    app_state["ready"] = True

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    # We run the setup in a separate thread or just block startup if it's critical.
    # For simplicity in this example, we'll run it directly, but keep in mind 
    # large uploads might block startup time.
    logger.info("Starting up RAG system...")
    setup_rag_system()
    yield
    # Shutdown logic (optional: delete store)
    logger.info("Shutting down...")
    # If you wanted to clean up:
    # if app_state["client"] and app_state["store_name"]:
    #     app_state["client"].file_search_stores.delete(name=app_state["store_name"])

app = FastAPI(lifespan=lifespan, title="Gemini RAG API")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str] = []

@app.get("/health")
def health_check():
    return {"status": "ok", "rag_ready": app_state["ready"]}

@app.post("/query", response_model=QueryResponse)
def query_model(request: QueryRequest):
    if not app_state["ready"]:
        raise HTTPException(status_code=503, detail="RAG system is not ready yet. Please wait.")
    
    client = app_state["client"]
    store_name = app_state["store_name"]
    
    if not client or not store_name:
         raise HTTPException(status_code=500, detail="RAG system configuration failed.")

    try:
        # Define the tool configuration
        tool = types.Tool(
            file_search=types.FileSearch(
                file_search_store_names=[store_name]
            )
        )
        
        generate_config = types.GenerateContentConfig(
            tools=[tool],
            temperature=0.7,
            system_instruction="You are a friendly and helpful AI assistant. Your goal is to answer the user's questions naturally and conversationally, as if you were a human expert explaining the topic. Use the provided context to answer accurately, but avoid sounding like a robot or just listing facts. Engage with the user."
        )
        
        logger.info(f"Processing query: {request.query}")
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=request.query,
            config=generate_config
        )
        
        answer = response.text
        sources = []
        
        # Extract citations if available
        if response.candidates:
            cand = response.candidates[0]
            if cand.grounding_metadata and cand.grounding_metadata.grounding_chunks:
                # Just collecting simple info for the response
                sources = [f"Source {i+1}" for i, _ in enumerate(cand.grounding_metadata.grounding_chunks)]

        return QueryResponse(answer=answer, sources=sources)

    except Exception as e:
        logger.error(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Determine port dynamically or default to 8000
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
