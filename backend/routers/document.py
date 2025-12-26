from datetime import datetime, timezone
from fastapi import APIRouter, File, UploadFile, Depends, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from dotenv import dotenv_values
import os
import shutil
import asyncio
from bson import ObjectId
from pydantic_models import document_schemas
from .utils import (
    verify_token, 
    convert_to_md, 
    detect_ai_generated_content, 
    detect_similarity, 
    read_md_file, 
    scrape_and_save_research_papers,
    build_search_query,
    get_similarity_threshold
)
from .logger import logger
from database import document_collection

config = dotenv_values(".env")
router = APIRouter(prefix="/document", tags=["document"])

# Folder where uploaded documents are stored.
DOCUMENTS_FOLDER = os.path.join(os.path.dirname(__file__), "..", "documents")
os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)

# Folder for scraped paper markdown files.
# Assuming scraped_papers folder already exists at backend root.
SCRAPED_FOLDER = os.path.join(os.path.dirname(__file__), "..", "scraped_papers")
os.makedirs(SCRAPED_FOLDER, exist_ok=True)

# ---------------------------------------------------------------------------
# LIGHTWEIGHT UPLOAD (NO PROCESSING)
# ---------------------------------------------------------------------------
@router.post("/upload/light")
async def upload_document_light(
    token_data: dict = Depends(verify_token),
    document: UploadFile = File(...),
    doc_type: str = Form(...)
):
    """
    Uploads the document and saves basic metadata, without processing.
    """
    logger.info("Starting lightweight document upload process")
    try:
        file_path = os.path.join(DOCUMENTS_FOLDER, document.filename)
        content = await document.read()
        with open(file_path, "wb") as f:
            f.write(content)
        logger.info("File saved successfully")
    except Exception as e:
        logger.error(f"Error while saving file: {e}")
        return JSONResponse(content={"error": f"Failed to save file: {e}"}, status_code=500)

    try:
        user_id = token_data.get("user_id")
        new_doc = await document_collection.insert_one({
            "name": document.filename,
            "path": file_path,
            "doc_type": doc_type,
            "upload_date": datetime.now(timezone.utc),
            "user_id": ObjectId(user_id),
            "md_path": None,             # To be set after processing
            "ai_content_result": [],     # To be set after processing
            "similarity_result": []      # To be set after processing
        })
        db_doc = await document_collection.find_one({"_id": new_doc.inserted_id})
        logger.info("Document uploaded and saved to DB (light)")
        return {
            "document_id": str(db_doc["_id"]),
            "name": db_doc["name"],
            "path": db_doc["path"]
        }
    except Exception as e:
        logger.error(f"Error while creating DB record: {e}")
        return JSONResponse(content={"error": f"Failed to create DB record: {e}"}, status_code=500)

# ---------------------------------------------------------------------------
# PROCESS DOCUMENT (AFTER FINAL SUBMIT)
# ---------------------------------------------------------------------------
@router.post("/process/{document_id}")
async def process_document(document_id: str, token_data: dict = Depends(verify_token)):
    """
    Processes the uploaded document:
    1. Converts the uploaded PDF to Markdown and moves it to a standardized location.
    2. Reads the Markdown file to extract the title.
    3. Scrapes research papers using the title.
    4. Converts each scraped paper sequentially to Markdown and moves it to the SCRAPED_FOLDER.
    5. Computes AI-generated content and similarity scores using the uploaded documents Markdown
       and the scraped papers Markdown files.
    6. Updates the document record with the results.
    """
    logger.info(f"Processing document ID: {document_id}")
    document = await document_collection.find_one({"_id": ObjectId(document_id)})
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    file_path = document["path"]

    # Step 1: Convert uploaded PDF to Markdown.
    try:
        # Assuming convert_to_md is async.
        md_file_path = await convert_to_md(file_path)
        standardized_md_path = os.path.join(DOCUMENTS_FOLDER, os.path.basename(md_file_path))
        if os.path.exists(standardized_md_path):
            os.remove(standardized_md_path)
        shutil.move(md_file_path, standardized_md_path)
        logger.info("Markdown conversion complete")
    except Exception as e:
        logger.error(f"Markdown conversion failed: {e}")
        return JSONResponse(content={"error": f"Markdown conversion failed: {e}"}, status_code=500)

    # Step 2: Read Markdown file and extract title + build search query.
    try:
        md_content = await asyncio.to_thread(read_md_file, standardized_md_path)
        title = md_content.split("\n")[0].replace("#", "").strip()
        logger.info("Title extracted from Markdown: %s", title)
        
        # Build optimized search query from keywords
        search_query = build_search_query(md_content, title)
    except Exception as e:
        logger.error(f"Error reading Markdown file: {e}")
        return JSONResponse(content={"error": f"Failed to read Markdown: {e}"}, status_code=500)

    # Step 3: Scrape research papers using keyword-based search.
    try:
        # Get document type for threshold configuration
        doc_type = document.get("doc_type", "default")
        similarity_threshold = get_similarity_threshold(doc_type)
        
        # Use keyword-based search instead of just title
        scraped_paper_details = await scrape_and_save_research_papers(search_query, max_results=5)
        logger.info("Scraped %d papers", len(scraped_paper_details))
    except Exception as e:
        logger.error(f"Error during scraping: {e}")
        return JSONResponse(content={"error": f"Scraping failed: {e}"}, status_code=500)

    # Step 4: Convert each scraped paper to Markdown sequentially.
    try:
        for paper in scraped_paper_details:
            logger.info("Converting scraped paper %s to Markdown", paper["path"])
            # Since convert_to_md is async, await directly.
            converted_md_path = await convert_to_md(paper["path"])
            # Move the converted file to SCRAPED_FOLDER.
            target_path = os.path.join(SCRAPED_FOLDER, os.path.basename(converted_md_path))
            if os.path.exists(target_path):
                os.remove(target_path)
            shutil.move(converted_md_path, target_path)
            paper["md_path"] = target_path
            logger.info("Scraped paper converted and saved to: %s", target_path)
        logger.info("All scraped papers converted to Markdown sequentially")
    except Exception as e:
        logger.error(f"Error converting scraped papers: {e}")
        return JSONResponse(content={"error": f"Failed to convert scraped papers: {e}"}, status_code=500)

    # Step 5: Compute AI and similarity scores.
    try:
        # If detect_ai_generated_content is synchronous (returns a list), wrap with asyncio.to_thread.
        ai_score = await asyncio.to_thread(detect_ai_generated_content, standardized_md_path)
        # Offload similarity detection concurrently with configurable threshold.
        similarity_tasks = [
            asyncio.to_thread(detect_similarity, standardized_md_path, paper["md_path"], paper, similarity_threshold)
            for paper in scraped_paper_details
        ]
        text_similarity_scores = await asyncio.gather(*similarity_tasks, return_exceptions=True)
        logger.info("AI and similarity computations complete")
    except Exception as e:
        logger.error(f"Error during AI/similarity computation: {e}")
        return JSONResponse(content={"error": f"Failed to compute AI and similarity scores: {e}"}, status_code=500)

    # Step 6: Update the document record with processed data.
    try:
        await document_collection.update_one(
            {"_id": ObjectId(document_id)},
            {"$set": {
                "md_path": standardized_md_path,
                "ai_content_result": [ai.dict() for ai in ai_score] if isinstance(ai_score, list) else ai_score,
                "similarity_result": text_similarity_scores
            }}
        )
        logger.info("Document record updated with processing results")
        return {"message": "Processing completed", "document_id": document_id}
    except Exception as e:
        logger.error(f"Error updating document record: {e}")
        return JSONResponse(content={"error": f"Failed to update document: {e}"}, status_code=500)

# ---------------------------------------------------------------------------
# GET Endpoints for Documents (unchanged)
# ---------------------------------------------------------------------------
@router.get("/")
async def get_all_documents(token_data: dict = Depends(verify_token)):
    try:
        user_id = token_data.get("user_id")
        documents = []
        cursor = document_collection.find({"user_id": ObjectId(user_id)})
        async for doc in cursor:
            documents.append(document_schemas.Document(**doc))
        return document_schemas.DocumentResponse(documents=documents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {e}")

@router.get("/{document_id}")
async def get_document(document_id: str, token_data: dict = Depends(verify_token)):
    document = await document_collection.find_one({"_id": ObjectId(document_id)})
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    document["_id"] = str(document["_id"])
    document["user_id"] = str(document["user_id"])
    return document_schemas.Document(**document)

@router.get("/file/{filename}")
async def get_document_by_filename(filename: str):
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DOCUMENTS_DIR = os.path.join(ROOT_DIR, "documents")
    file_path = os.path.join(DOCUMENTS_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        raise HTTPException(status_code=404, detail="File not found")

@router.post("/upload")
async def upload_document(token_data: dict = Depends(verify_token), document: UploadFile = File(...)):
    logger.info("Starting document upload process")
    
    if not os.path.exists(DOCUMENTS_FOLDER):
        logger.info("Creating documents folder")
        os.makedirs(DOCUMENTS_FOLDER)

    try:
        file_path = os.path.join(DOCUMENTS_FOLDER, document.filename)
        logger.info("Saving file...")
        content = await document.read()
        with open(file_path, "wb") as f:
            f.write(content)
        logger.info("File saved successfully")
    except Exception as e:
        logger.error(f"Error while saving file: {e}")
        return JSONResponse(content={"error": f"Failed to save file: {e}"}, status_code=500)

    try:
        logger.info("Converting file to markdown...")
        md_file_path = await convert_to_md(file_path)
        logger.info("Converted file to markdown")
        standardized_md_path = os.path.join(DOCUMENTS_FOLDER, os.path.basename(md_file_path))
        if os.path.exists(standardized_md_path):
            logger.info("Removing existing markdown file...")
            os.remove(standardized_md_path)
        shutil.move(md_file_path, standardized_md_path)
        logger.info("Markdown file moved to standardized location")
    except ValueError as e:
        logger.error(f"Markdown conversion error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=400)
    except Exception as e:
        logger.error(f"Markdown conversion failed: {e}")
        return JSONResponse(content={"error": f"Markdown conversion failed: {e}"}, status_code=500)

    try:
        logger.info("Reading markdown file...")
        md_content = await asyncio.to_thread(read_md_file, standardized_md_path)
        title = md_content.split("\n")[0].replace("#", "").strip()
        logger.info("Extracted title from markdown: %s", title)
        logger.info("Scraping papers from ArXiv...")
        scraped_paper_details = await scrape_and_save_research_papers(title)
        logger.info("Scraped %d papers", len(scraped_paper_details))
        for paper in scraped_paper_details:
            logger.info("Converting scraped paper %s to markdown sequentially", paper["path"])
            md_path = await convert_to_md(paper["path"])
            target_path = os.path.join(SCRAPED_FOLDER, os.path.basename(md_path))
            if os.path.exists(target_path):
                os.remove(target_path)
            shutil.move(md_path, target_path)
            paper["md_path"] = target_path
            logger.info("Scraped paper converted and saved to: %s", target_path)
        logger.info("All scraped papers converted to markdown sequentially")
    except Exception as e:
        logger.error(f"Error during scraping: {e}")
        return JSONResponse(content={"error": f"Failed to scrape papers: {e}"}, status_code=500)

    results = []
    try:
        logger.info("Starting AI and similarity score computation")
        # Wrap synchronous AI detection and similarity calls.
        ai_score = await asyncio.to_thread(detect_ai_generated_content, standardized_md_path)
        similarity_tasks = [
            asyncio.to_thread(detect_similarity, standardized_md_path, paper["md_path"], paper)
            for paper in scraped_paper_details
        ]
        text_similarity_scores = await asyncio.gather(*similarity_tasks, return_exceptions=True)
        results.append({
            "ai_score": ai_score,
            "text_similarity_scores": text_similarity_scores
        })
        logger.info("AI and similarity score computation completed")
    except Exception as e:
        logger.error(f"Error during AI and similarity score computation: {e}")
        return JSONResponse(content={"error": f"Failed to compute AI and similarity scores: {e}"}, status_code=500)

    try:
        logger.info("Creating DB record for the uploaded document")
        ai_score_dict = [ai.dict() for ai in ai_score] if isinstance(ai_score, list) else ai_score
        similarity_score_dict = text_similarity_scores
        user_id = token_data.get("user_id")
        new_document = await document_collection.insert_one({
            "name": document.filename,
            "path": file_path,
            "md_path": standardized_md_path,
            "ai_content_result": ai_score_dict,
            "similarity_result": similarity_score_dict,
            "upload_date": datetime.now(timezone.utc),
            "user_id": ObjectId(user_id)
        })
        db_document = await document_collection.find_one({"_id": new_document.inserted_id})
        logger.info("Document uploaded and saved to DB successfully")
        return document_schemas.Document.model_validate(db_document)
    except Exception as e:
        logger.error(f"Error while creating DB record: {e}")
        return JSONResponse(content={"error": f"Failed to create DB record: {e}"}, status_code=500)
