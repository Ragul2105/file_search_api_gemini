import os
import time
import glob
import concurrent.futures
import logging
from contextlib import asynccontextmanager
from typing import List, Optional
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from pydantic import BaseModel
from google import genai
from google.genai import types

# Load environment variables from .env file
load_dotenv()

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
    """Initializes the RAG system: reuses existing store or creates new one."""
    api_key = get_api_key()
    if not api_key:
        logger.error("API Key missing. RAG system setup failed.")
        return

    client = genai.Client(api_key=api_key)
    app_state["client"] = client
    logger.info("Client initialized.")

    # Check if we have an existing store name in environment
    existing_store_name = os.environ.get("FILE_SEARCH_STORE_NAME")
    
    if existing_store_name:
        try:
            # Try to get the existing store
            logger.info(f"Checking for existing store: {existing_store_name}")
            file_search_store = client.file_search_stores.get(name=existing_store_name)
            app_state["store_name"] = file_search_store.name
            logger.info(f"Reusing existing store: {file_search_store.name}")
            logger.info(f"Store has {file_search_store.active_documents_count} active documents")
        except Exception as e:
            logger.warning(f"Could not find existing store: {e}")
            logger.info("Creating new store...")
            existing_store_name = None
    
    # Create new store if we don't have one or couldn't find the existing one
    if not existing_store_name:
        try:
            file_search_store = client.file_search_stores.create(
                config={'display_name': 'fastapi_knowledge_base'}
            )
            app_state["store_name"] = file_search_store.name
            logger.info(f"Created new store: {file_search_store.name}")
            logger.info(f"⚠️  To reuse this store on restart, add to .env file:")
            logger.info(f"FILE_SEARCH_STORE_NAME={file_search_store.name}")
        except Exception as e:
            logger.error(f"Failed to create store: {e}")
            return

    # Mark system as ready - files will be uploaded via /upload API
    logger.info("RAG system ready. Use /upload endpoint to add documents.")
    app_state["ready"] = True

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic - only creates the store, no file uploads
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

class UploadResponse(BaseModel):
    success: bool
    message: str
    file_name: Optional[str] = None

class DocumentInfo(BaseModel):
    name: str
    display_name: str
    size_bytes: Optional[int] = None
    mime_type: Optional[str] = None
    state: Optional[str] = None
    create_time: Optional[str] = None
    update_time: Optional[str] = None
    
    class Config:
        json_encoders = {
            int: str
        }

class ListFilesResponse(BaseModel):
    documents: List[DocumentInfo]
    total_count: int

class DeleteFileResponse(BaseModel):
    success: bool
    message: str
    document_name: str

class StoreInfo(BaseModel):
    name: str
    display_name: str
    active_documents_count: Optional[int] = None
    pending_documents_count: Optional[int] = None
    failed_documents_count: Optional[int] = None
    size_bytes: Optional[int] = None
    create_time: Optional[str] = None
    update_time: Optional[str] = None

class ListStoresResponse(BaseModel):
    stores: List[StoreInfo]
    total_count: int
    current_store: Optional[str] = None

class DeleteStoreResponse(BaseModel):
    success: bool
    message: str
    store_name: str

@app.get("/health")
def health_check():
    return {"status": "ok", "rag_ready": app_state["ready"]}

@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file to the File Search store.
    Strictly follows Google's File Search documentation.
    """
    if not app_state["ready"]:
        raise HTTPException(status_code=503, detail="RAG system is not ready yet. Please wait.")
    
    client = app_state["client"]
    store_name = app_state["store_name"]
    
    if not client or not store_name:
        raise HTTPException(status_code=500, detail="RAG system configuration failed.")
    
    # Save uploaded file temporarily
    temp_file_path = os.path.join("/tmp", file.filename)
    try:
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"Uploading file: {file.filename} to store: {store_name}")
        
        # Strictly following documentation: upload_to_file_search_store
        operation = client.file_search_stores.upload_to_file_search_store(
            file=temp_file_path,
            file_search_store_name=store_name,
            config={'display_name': file.filename}
        )
        
        logger.info(f"Upload operation initiated: {operation.name}")
        
        # Poll for completion as per documentation
        while not operation.done:
            time.sleep(2)
            operation = client.operations.get(operation)
        
        # Check for errors
        if operation.error:
            logger.error(f"Error processing {file.filename}: {operation.error}")
            raise HTTPException(status_code=500, detail=f"Upload failed: {operation.error}")
        
        logger.info(f"Successfully uploaded: {file.filename}")
        
        return UploadResponse(
            success=True,
            message=f"File {file.filename} uploaded successfully",
            file_name=file.filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Exception during upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.get("/files/list", response_model=ListFilesResponse)
def list_files():
    """
    List all documents in the File Search store.
    Strictly follows Google's File Search documentation.
    """
    if not app_state["ready"]:
        raise HTTPException(status_code=503, detail="RAG system is not ready yet. Please wait.")
    
    client = app_state["client"]
    store_name = app_state["store_name"]
    
    if not client or not store_name:
        raise HTTPException(status_code=500, detail="RAG system configuration failed.")
    
    try:
        logger.info(f"Listing documents in store: {store_name}")
        
        # Strictly following documentation: list documents
        documents = []
        for document in client.file_search_stores.documents.list(parent=store_name):
            documents.append(DocumentInfo(
                name=document.name,
                display_name=document.display_name or "Unknown",
                size_bytes=document.size_bytes,
                mime_type=document.mime_type,
                state=document.state,
                create_time=str(document.create_time) if document.create_time else None,
                update_time=str(document.update_time) if document.update_time else None
            ))
        
        logger.info(f"Found {len(documents)} documents in store")
        
        return ListFilesResponse(
            documents=documents,
            total_count=len(documents)
        )
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stores/list", response_model=ListStoresResponse)
def list_stores():
    """
    List all File Search stores associated with the account.
    Strictly follows Google's File Search documentation.
    """
    if not app_state["client"]:
        raise HTTPException(status_code=500, detail="Client not initialized.")
    
    client = app_state["client"]
    
    try:
        logger.info("Listing all file search stores...")
        
        # Strictly following documentation: list all stores
        stores = []
        for store in client.file_search_stores.list():
            stores.append(StoreInfo(
                name=store.name,
                display_name=store.display_name or "Unknown",
                active_documents_count=store.active_documents_count,
                pending_documents_count=store.pending_documents_count,
                failed_documents_count=store.failed_documents_count,
                size_bytes=store.size_bytes,
                create_time=str(store.create_time) if store.create_time else None,
                update_time=str(store.update_time) if store.update_time else None
            ))
        
        logger.info(f"Found {len(stores)} stores")
        
        return ListStoresResponse(
            stores=stores,
            total_count=len(stores),
            current_store=app_state.get("store_name")
        )
        
    except Exception as e:
        logger.error(f"Error listing stores: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stores/current", response_model=StoreInfo)
def get_current_store():
    """
    Get information about the currently active File Search store.
    Strictly follows Google's File Search documentation.
    """
    if not app_state["ready"]:
        raise HTTPException(status_code=503, detail="RAG system is not ready yet. Please wait.")
    
    client = app_state["client"]
    store_name = app_state["store_name"]
    
    if not client or not store_name:
        raise HTTPException(status_code=500, detail="RAG system configuration failed.")
    
    try:
        logger.info(f"Getting info for current store: {store_name}")
        
        # Strictly following documentation: get store by name
        store = client.file_search_stores.get(name=store_name)
        
        return StoreInfo(
            name=store.name,
            display_name=store.display_name or "Unknown",
            active_documents_count=store.active_documents_count,
            pending_documents_count=store.pending_documents_count,
            failed_documents_count=store.failed_documents_count,
            size_bytes=store.size_bytes,
            create_time=str(store.create_time) if store.create_time else None,
            update_time=str(store.update_time) if store.update_time else None
        )
        
    except Exception as e:
        logger.error(f"Error getting store info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/files/delete/{document_id}", response_model=DeleteFileResponse)
def delete_file(document_id: str):
    """
    Delete a specific document from the File Search store.
    Strictly follows Google's File Search documentation.
    
    Args:
        document_id: The document ID (not the full resource name)
                    Example: 'the-doc-abc' from 'fileSearchStores/store-123/documents/the-doc-abc'
    """
    if not app_state["ready"]:
        raise HTTPException(status_code=503, detail="RAG system is not ready yet. Please wait.")
    
    client = app_state["client"]
    store_name = app_state["store_name"]
    
    if not client or not store_name:
        raise HTTPException(status_code=500, detail="RAG system configuration failed.")
    
    try:
        # Construct the full document name as per documentation
        # Format: fileSearchStores/{store}/documents/{document}
        document_name = f"{store_name}/documents/{document_id}"
        
        logger.info(f"Deleting document: {document_name}")
        
        # Strictly following documentation: delete document with force=True
        # force=True will delete the document along with its chunks
        client.file_search_stores.documents.delete(
            name=document_name,
            config={'force': True}
        )
        
        logger.info(f"Successfully deleted document: {document_name}")
        
        return DeleteFileResponse(
            success=True,
            message=f"Document {document_id} deleted successfully",
            document_name=document_name
        )
        
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/stores/delete/{store_id}", response_model=DeleteStoreResponse)
def delete_store(store_id: str):
    """
    Delete a File Search store.
    
    Args:
        store_id: The ID of the store (e.g., 'fastapiknowledgebase-32kyzpqzre6x')
                 Do NOT include 'fileSearchStores/' prefix.
    
    Note: This will delete the store and ALL its documents.
          Use force=True to ensure complete deletion.
    """
    if not app_state["ready"]:
        raise HTTPException(status_code=503, detail="RAG system is not ready yet.")
    
    client = app_state["client"]
    
    if not client:
        raise HTTPException(status_code=500, detail="RAG system configuration failed.")
    
    try:
        # Construct the full store name as per documentation
        # Format: fileSearchStores/{store_id}
        full_store_name = f"fileSearchStores/{store_id}"
        
        logger.info(f"Deleting store: {full_store_name}")
        
        # Strictly following documentation: delete store with force=True
        # force=True will delete the store along with all its documents
        client.file_search_stores.delete(
            name=full_store_name,
            config={'force': True}
        )
        
        logger.info(f"Successfully deleted store: {full_store_name}")
        
        # If the deleted store is the current one, clear the app state
        if app_state.get("store_name") == full_store_name:
            logger.warning("Deleted the current active store. Please restart the server.")
            app_state["store_name"] = None
            app_state["ready"] = False
        
        return DeleteStoreResponse(
            success=True,
            message=f"Store {store_id} and all its documents deleted successfully",
            store_name=full_store_name
        )
        
    except Exception as e:
        logger.error(f"Error deleting store {store_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
