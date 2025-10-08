# main (4).py - UPDATED VERSION
# main.py - COMPLETE UPDATED VERSION WITH FIXED DATABASE IMPORT
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd
import io
import numpy as np
import tempfile
import os
from typing import Optional
import json
import uvicorn
import shutil
from backend import (
    run_fast_pipeline, 
    run_config1_pipeline, 
    run_deep_pipeline, 
    retrieve_similar,
    export_chunks,
    export_embeddings,
    get_system_info,
    get_file_info,
    connect_mysql,
    connect_postgresql,
    get_table_list,
    import_table_to_dataframe,
    process_large_file,
    can_load_file,
    LARGE_FILE_THRESHOLD,
    process_file_direct,
    EMBEDDING_BATCH_SIZE,
    PARALLEL_WORKERS,
    export_preprocessed_data  # NEW: Added export function
)

app = FastAPI(title="Chunking Optimizer API", version="2.0")

# ---------------------------
# OpenAI-compatible API Endpoints
# ---------------------------
@app.post("/v1/embeddings")
async def openai_embeddings(
    model: str = Form("text-embedding-ada-002"),
    input: str = Form(...),
    openai_api_key: Optional[str] = Form(None),
    openai_base_url: Optional[str] = Form(None)
):
    """OpenAI-compatible embeddings endpoint"""
    try:
        from backend import OpenAIEmbeddingAPI
        
        embedding_api = OpenAIEmbeddingAPI(
            model_name=model,
            api_key=openai_api_key,
            base_url=openai_base_url
        )
        
        # Handle both string and list of strings
        if isinstance(input, str):
            texts = [input]
        else:
            texts = input
            
        embeddings = embedding_api.encode(texts)
        
        # Format response in OpenAI standard
        response_data = {
            "object": "list",
            "data": [],
            "model": model,
            "usage": {
                "prompt_tokens": sum(len(text.split()) for text in texts),
                "total_tokens": sum(len(text.split()) for text in texts)
            }
        }
        
        for i, embedding in enumerate(embeddings):
            response_data["data"].append({
                "object": "embedding",
                "embedding": embedding.tolist(),
                "index": i
            })
            
        return JSONResponse(content=response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")

@app.post("/v1/chat/completions")
async def openai_chat_completions(
    model: str = Form("gpt-3.5-turbo"),
    messages: str = Form(...),
    max_tokens: Optional[int] = Form(1000),
    temperature: Optional[float] = Form(0.7),
    openai_api_key: Optional[str] = Form(None),
    openai_base_url: Optional[str] = Form(None)
):
    """OpenAI-compatible chat completions endpoint (requires external OpenAI API)"""
    try:
        import openai
        
        if openai_api_key:
            openai.api_key = openai_api_key
        if openai_base_url:
            openai.base_url = openai_base_url
            
        # Parse messages from JSON string
        messages_list = json.loads(messages)
        
        response = openai.chat.completions.create(
            model=model,
            messages=messages_list,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return JSONResponse(content=response.dict())
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat completion error: {str(e)}")

# ---------------------------
# Enhanced DB IMPORT with Large File Support (FIXED)
# ---------------------------
@app.post("/db/test_connection")
async def db_test_connection(
    db_type: str = Form(...),
    host: str = Form(...),
    port: int = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    database: str = Form(...)
):
    try:
        if db_type == "mysql":
            conn = connect_mysql(host, port, username, password, database)
        elif db_type == "postgresql":
            conn = connect_postgresql(host, port, username, password, database)
        else:
            return {"status": "error", "message": "Unsupported db_type"}
        conn.close()
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/db/list_tables")
async def db_list_tables(
    db_type: str = Form(...),
    host: str = Form(...),
    port: int = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    database: str = Form(...)
):
    try:
        if db_type == "mysql":
            conn = connect_mysql(host, port, username, password, database)
        elif db_type == "postgresql":
            conn = connect_postgresql(host, port, username, password, database)
        else:
            return {"error": "Unsupported db_type"}
        tables = get_table_list(conn, db_type)
        conn.close()
        
        # Filter out system tables
        if db_type == "postgresql":
            # Filter out PostgreSQL system tables
            system_tables = ['pg_', 'sql_', 'information_schema', 'system_']
            tables = [table for table in tables if not any(table.startswith(prefix) for prefix in system_tables)]
        elif db_type == "mysql":
            # Filter out MySQL system tables
            system_tables = ['mysql', 'information_schema', 'performance_schema', 'sys']
            tables = [table for table in tables if table not in system_tables]
        
        return {"tables": tables}
    except Exception as e:
        return {"error": str(e)}

# ---------------------------
# Enhanced Pipeline Endpoints with Large File Support
# ---------------------------
@app.post("/run_fast")
async def run_fast(
    file: Optional[UploadFile] = File(None),
    db_type: Optional[str] = Form("sqlite"),
    host: Optional[str] = Form(None),
    port: Optional[int] = Form(None),
    username: Optional[str] = Form(None),
    password: Optional[str] = Form(None),
    database: Optional[str] = Form(None),
    table_name: Optional[str] = Form(None),
    use_openai: bool = Form(False),
    openai_api_key: Optional[str] = Form(None),
    openai_base_url: Optional[str] = Form(None),
    process_large_files: bool = Form(True),
    use_turbo: bool = Form(True),
    batch_size: int = Form(256)
):
    try:
        # Handle database import
        if host and port and username and password and database and table_name:
            if db_type == "mysql":
                conn = connect_mysql(host, port, username, password, database)
            elif db_type == "postgresql":
                conn = connect_postgresql(host, port, username, password, database)
            else:
                raise HTTPException(status_code=400, detail="Unsupported database type")
            
            df = import_table_to_dataframe(conn, table_name)
            conn.close()
            
            file_info = {
                "filename": f"{table_name}_from_{database}",
                "file_size": len(df),
                "upload_time": "Database import",
                "location": "Database"
            }
            
        # Handle file upload
        elif file:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                shutil.copyfileobj(file.file, tmp_file)
                temp_path = tmp_file.name
            
            file_info = {
                "filename": file.filename,
                "file_size": os.path.getsize(temp_path),
                "upload_time": "File upload",
                "location": "Temporary storage"
            }
            
            # Check if we should process as large file
            if process_large_files and not can_load_file(file_info['file_size']):
                # Use direct file processing for large files
                result = process_file_direct(
                    temp_path,
                    processing_mode="fast",
                    use_openai=use_openai,
                    openai_api_key=openai_api_key,
                    openai_base_url=openai_base_url,
                    use_turbo=use_turbo,
                    batch_size=batch_size
                )
                
                # Clean up temporary file
                os.unlink(temp_path)
                
                return {
                    "mode": "fast",
                    "summary": result,
                    "large_file_processed": True
                }
            else:
                # Load into memory for normal processing
                df = pd.read_csv(temp_path)
                os.unlink(temp_path)  # Clean up temp file
            
        else:
            raise HTTPException(status_code=400, detail="No file or database configuration provided")
        
        # Run pipeline
        result = run_fast_pipeline(
            df,
            db_type=db_type,
            db_config=None,
            file_info=file_info,
            use_openai=use_openai,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            use_turbo=use_turbo,
            batch_size=batch_size
        )
        
        return {
            "mode": "fast",
            "summary": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fast pipeline error: {str(e)}")

@app.post("/run_config1")
async def run_config1(
    file: Optional[UploadFile] = File(None),
    db_type: Optional[str] = Form("sqlite"),
    host: Optional[str] = Form(None),
    port: Optional[int] = Form(None),
    username: Optional[str] = Form(None),
    password: Optional[str] = Form(None),
    database: Optional[str] = Form(None),
    table_name: Optional[str] = Form(None),
    null_handling: str = Form("keep"),
    fill_value: Optional[str] = Form("Unknown"),
    chunk_method: str = Form("recursive"),
    chunk_size: int = Form(400),
    overlap: int = Form(50),
    model_choice: str = Form("paraphrase-MiniLM-L6-v2"),
    storage_choice: str = Form("faiss"),
    use_openai: bool = Form(False),
    openai_api_key: Optional[str] = Form(None),
    openai_base_url: Optional[str] = Form(None),
    process_large_files: bool = Form(True),
    use_turbo: bool = Form(True),
    batch_size: int = Form(256)
):
    try:
        # Handle database import
        if host and port and username and password and database and table_name:
            if db_type == "mysql":
                conn = connect_mysql(host, port, username, password, database)
            elif db_type == "postgresql":
                conn = connect_postgresql(host, port, username, password, database)
            else:
                raise HTTPException(status_code=400, detail="Unsupported database type")
            
            df = import_table_to_dataframe(conn, table_name)
            conn.close()
            
            file_info = {
                "filename": f"{table_name}_from_{database}",
                "file_size": len(df),
                "upload_time": "Database import",
                "location": "Database"
            }
            
        # Handle file upload
        elif file:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                shutil.copyfileobj(file.file, tmp_file)
                temp_path = tmp_file.name
            
            file_info = {
                "filename": file.filename,
                "file_size": os.path.getsize(temp_path),
                "upload_time": "File upload",
                "location": "Temporary storage"
            }
            
            # Check if we should process as large file
            if process_large_files and not can_load_file(file_info['file_size']):
                # Use direct file processing for large files
                config = {
                    "null_handling": null_handling,
                    "fill_value": fill_value,
                    "chunk_method": chunk_method,
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "model_choice": model_choice,
                    "storage_choice": storage_choice,
                }
                
                result = process_file_direct(
                    temp_path,
                    processing_mode="config1",
                    **config,
                    use_openai=use_openai,
                    openai_api_key=openai_api_key,
                    openai_base_url=openai_base_url,
                    use_turbo=use_turbo,
                    batch_size=batch_size
                )
                
                # Clean up temporary file
                os.unlink(temp_path)
                
                return {
                    "mode": "config1",
                    "summary": result,
                    "large_file_processed": True
                }
            else:
                # Load into memory for normal processing
                df = pd.read_csv(temp_path)
                os.unlink(temp_path)  # Clean up temp file
            
        else:
            raise HTTPException(status_code=400, detail="No file or database configuration provided")
        
        # Run pipeline
        result = run_config1_pipeline(
            df,
            null_handling=null_handling,
            fill_value=fill_value,
            chunk_method=chunk_method,
            chunk_size=chunk_size,
            overlap=overlap,
            model_choice=model_choice,
            storage_choice=storage_choice,
            db_config=None,
            file_info=file_info,
            use_openai=use_openai,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            use_turbo=use_turbo,
            batch_size=batch_size
        )
        
        return {
            "mode": "config1",
            "summary": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Config1 pipeline error: {str(e)}")

@app.post("/run_deep")
async def run_deep(
    file: Optional[UploadFile] = File(None),
    db_type: Optional[str] = Form("sqlite"),
    host: Optional[str] = Form(None),
    port: Optional[int] = Form(None),
    username: Optional[str] = Form(None),
    password: Optional[str] = Form(None),
    database: Optional[str] = Form(None),
    table_name: Optional[str] = Form(None),
    null_handling: str = Form("keep"),
    fill_value: Optional[str] = Form("Unknown"),
    remove_stopwords: bool = Form(False),
    lowercase: bool = Form(True),
    text_processing_option: str = Form("none"),
    chunk_method: str = Form("recursive"),
    chunk_size: int = Form(400),
    overlap: int = Form(50),
    model_choice: str = Form("paraphrase-MiniLM-L6-v2"),
    storage_choice: str = Form("faiss"),
    column_types: str = Form("{}"),
    document_key_column: Optional[str] = Form(None),
    use_openai: bool = Form(False),
    openai_api_key: Optional[str] = Form(None),
    openai_base_url: Optional[str] = Form(None),
    process_large_files: bool = Form(True),
    use_turbo: bool = Form(True),
    batch_size: int = Form(256)
):
    try:
        # Parse column types
        try:
            column_types_dict = json.loads(column_types) if column_types else {}
        except:
            column_types_dict = {}
        
        # Handle database import
        if host and port and username and password and database and table_name:
            if db_type == "mysql":
                conn = connect_mysql(host, port, username, password, database)
            elif db_type == "postgresql":
                conn = connect_postgresql(host, port, username, password, database)
            else:
                raise HTTPException(status_code=400, detail="Unsupported database type")
            
            df = import_table_to_dataframe(conn, table_name)
            conn.close()
            
            file_info = {
                "filename": f"{table_name}_from_{database}",
                "file_size": len(df),
                "upload_time": "Database import",
                "location": "Database"
            }
            
        # Handle file upload
        elif file:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                shutil.copyfileobj(file.file, tmp_file)
                temp_path = tmp_file.name
            
            file_info = {
                "filename": file.filename,
                "file_size": os.path.getsize(temp_path),
                "upload_time": "File upload",
                "location": "Temporary storage"
            }
            
            # Check if we should process as large file
            if process_large_files and not can_load_file(file_info['file_size']):
                # Use direct file processing for large files
                config = {
                    "null_handling": null_handling,
                    "fill_value": fill_value,
                    "remove_stopwords": remove_stopwords,
                    "lowercase": lowercase,
                    "text_processing_option": text_processing_option,
                    "chunk_method": chunk_method,
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "model_choice": model_choice,
                    "storage_choice": storage_choice,
                    "column_types": column_types_dict,
                    "document_key_column": document_key_column,
                }
                
                result = process_file_direct(
                    temp_path,
                    processing_mode="deep",
                    **config,
                    use_openai=use_openai,
                    openai_api_key=openai_api_key,
                    openai_base_url=openai_base_url,
                    use_turbo=use_turbo,
                    batch_size=batch_size
                )
                
                # Clean up temporary file
                os.unlink(temp_path)
                
                return {
                    "mode": "deep",
                    "summary": result,
                    "large_file_processed": True
                }
            else:
                # Load into memory for normal processing
                df = pd.read_csv(temp_path)
                os.unlink(temp_path)  # Clean up temp file
            
        else:
            raise HTTPException(status_code=400, detail="No file or database configuration provided")
        
        # Run pipeline
        result = run_deep_pipeline(
            df,
            null_handling=null_handling,
            fill_value=fill_value,
            remove_stopwords=remove_stopwords,
            lowercase=lowercase,
            text_processing_option=text_processing_option,
            chunk_method=chunk_method,
            chunk_size=chunk_size,
            overlap=overlap,
            model_choice=model_choice,
            storage_choice=storage_choice,
            column_types=column_types_dict,
            document_key_column=document_key_column,
            db_config=None,
            file_info=file_info,
            use_openai=use_openai,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            use_turbo=use_turbo,
            batch_size=batch_size
        )
        
        return {
            "mode": "deep",
            "summary": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deep pipeline error: {str(e)}")

# ---------------------------
# Retrieval Endpoints
# ---------------------------
@app.post("/retrieve")
async def retrieve(
    query: str = Form(...),
    k: int = Form(5)
):
    try:
        result = retrieve_similar(query, k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")

@app.post("/v1/retrieve")
async def openai_retrieve(
    query: str = Form(...),
    model: str = Form("all-MiniLM-L6-v2"),
    n_results: int = Form(5)
):
    try:
        result = retrieve_similar(query, n_results)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")

# ---------------------------
# Export Endpoints - UPDATED: Removed numpy format download
# ---------------------------
@app.get("/export/chunks")
async def export_chunks_endpoint():
    try:
        chunks_text = export_chunks()
        return JSONResponse(content={"chunks": chunks_text})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export error: {str(e)}")

@app.get("/export/embeddings_text")
async def export_embeddings_text():
    try:
        embeddings = export_embeddings()
        if embeddings is not None:
            # Convert to text format
            embeddings_text = "Embeddings Shape: " + str(embeddings.shape) + "\n\n"
            embeddings_text += "First 10 embeddings:\n"
            for i, emb in enumerate(embeddings[:10]):
                embeddings_text += f"Embedding {i}: {emb[:5]}...\n"  # Show first 5 dimensions
            return JSONResponse(content={"embeddings": embeddings_text})
        else:
            raise HTTPException(status_code=404, detail="No embeddings available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export error: {str(e)}")

# NEW: Export preprocessed data endpoint
@app.get("/export/preprocessed")
async def export_preprocessed_endpoint():
    try:
        preprocessed_text = export_preprocessed_data()
        return JSONResponse(content={"preprocessed_data": preprocessed_text})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export error: {str(e)}")

# ---------------------------
# Information Endpoints
# ---------------------------
@app.get("/system_info")
async def system_info():
    try:
        info = get_system_info()
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"System info error: {str(e)}")

@app.get("/file_info")
async def file_info():
    try:
        info = get_file_info()
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File info error: {str(e)}")

@app.get("/capabilities")
async def capabilities():
    return {
        "large_file_support": True,
        "max_file_size": "3GB+",
        "performance_features": {
            "turbo_mode": True,
            "batch_processing": True,
            "parallel_workers": PARALLEL_WORKERS,
            "embedding_batch_size": EMBEDDING_BATCH_SIZE
        },
        "embedding_models": ["paraphrase-MiniLM-L6-v2", "all-MiniLM-L6-v2", "text-embedding-ada-002"],
        "vector_stores": ["faiss", "chroma"],
        "chunking_methods": ["fixed", "recursive", "semantic", "document"],
        "processing_modes": ["fast", "config1", "deep"]
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "2.0"}

# ---------------------------
# Main Entry Point
# ---------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)