import os
import time
import glob
import concurrent.futures
from google import genai
from google.genai import types

# Configuration
DOCUMENTS_DIR = "/Users/macbookpro16_stic_admin/Documents/file_search_latest/documents"
API_KEY_ENV_VAR = "GOOGLE_API_KEY"

def get_api_key():
    """Retrieves the API key from the environment."""
    api_key = os.environ.get(API_KEY_ENV_VAR)
    if not api_key:
        print(f"Error: {API_KEY_ENV_VAR} environment variable not set.")
        print("Please export your API key: export GOOGLE_API_KEY='your_key_here'")
        exit(1)
    return api_key

def upload_and_monitor(client, file_path, store_name):
    """Uploads a single file to the store and monitors its status."""
    file_name = os.path.basename(file_path)
    print(f"Starting upload for: {file_name}")
    
    try:
        # strictly following docs: pass file path string and config as dict
        operation = client.file_search_stores.upload_to_file_search_store(
            file=file_path,
            file_search_store_name=store_name,
            config={'display_name': file_name}
        )
        
        print(f"Upload initiated for {file_name}. Operation: {operation.name}")
        
        # Poll for completion
        # Strictly following docs: pass the operation object itself to get()
        while not operation.done:
            time.sleep(2)
            operation = client.operations.get(operation)
            
        if operation.error:
            print(f"Error processing {file_name}: {operation.error}")
            return None
        
        print(f"Successfully processed: {file_name}")
        return file_name
        
    except Exception as e:
        print(f"Exception uploading {file_name}: {e}")
        return None

def main():
    api_key = get_api_key()
    
    # Initialize Client using the new SDK pattern
    client = genai.Client(api_key=api_key)
    print("Client initialized.")

    # 1. Create File Search Store
    print("Creating File Search Store...")
    try:
        # strictly following docs: config is a dict
        file_search_store = client.file_search_stores.create(
            config={'display_name': 'document_knowledge_base'}
        )
        print(f"Created store: {file_search_store.name} (Display Name: {file_search_store.display_name})")
    except Exception as e:
        print(f"Failed to create store: {e}")
        return

    # 2. List files
    pdf_files = glob.glob(os.path.join(DOCUMENTS_DIR, "*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {DOCUMENTS_DIR}")
        return
    
    print(f"Found {len(pdf_files)} PDF files.")

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

    print(f"\nUploads complete. {successful_uploads}/{len(pdf_files)} files processed successfully.")

    if successful_uploads == 0:
        print("No files were successfully uploaded. Exiting.")
        return

    # 4. Generate Content with File Search Tool
    # In the new SDK, we don't create a 'model' object the same way.
    # We use client.models.generate_content directly.
    
    model_id = "gemini-2.5-flash" 
    
    # Define the tool configuration
    tool = types.Tool(
        file_search=types.FileSearch(
            file_search_store_names=[file_search_store.name]
        )
    )
    
    generate_config = types.GenerateContentConfig(
        tools=[tool],
        temperature=0.5
    )

    # 5. Interactive Query Loop
    print("\n" + "="*50)
    print(f"RAG System Ready (Model: {model_id})! Type 'exit' to quit.")
    print("="*50)

    while True:
        user_input = input("\nQuery: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        
        if not user_input.strip():
            continue

        try:
            print("Thinking...")
            response = client.models.generate_content(
                model=model_id,
                contents=user_input,
                config=generate_config
            )
            
            print("\nResponse:")
            print(response.text)
            
            # Check for citations/grounding
            # The structure of response.candidates[0].grounding_metadata might vary
            # but usually follows the API definition.
            if response.candidates:
                cand = response.candidates[0]
                if cand.grounding_metadata and cand.grounding_metadata.grounding_chunks:
                    chunks = cand.grounding_metadata.grounding_chunks
                    print("\nSources used:")
                    print(f"Referenced {len(chunks)} passages from uploaded docs.")

        except Exception as e:
            print(f"Error generating response: {e}")

if __name__ == "__main__":
    main()