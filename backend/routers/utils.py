from typing import List, Optional
from fastapi import HTTPException, Request, status
from jose import JWTError, jwt
from jose.exceptions import ExpiredSignatureError
from dotenv import dotenv_values
from docx import Document as DocxDocument
import subprocess
import os
import re
from collections import Counter
from nlp.similarity_detection import research_similarity

from nlp.ai_detection.roberta_ai_detection import roberta_ai_detection
from nlp.ai_detection.detect_gpt_detection import detect_gpt_main
from pydantic_models.document_schemas import AIGeneratedContent, Similarity, SimilaritySource

from pathlib import Path
from paperscraper.pdf import save_pdf
from paperscraper.arxiv import get_arxiv_papers
from .logger import logger

config = dotenv_values(".env")

# ---------------------------------------------------------------------------
# CONFIGURABLE THRESHOLDS PER DOCUMENT TYPE
# ---------------------------------------------------------------------------
# Higher thresholds reduce false positives (require more similarity to flag)
# Lower thresholds are more sensitive (flag lower similarity)
SIMILARITY_THRESHOLDS = {
    "research_paper": 0.70,      # Academic papers have high natural similarity
    "thesis": 0.65,              # Theses may reference more sources
    "assignment": 0.60,          # Student work should be more original
    "report": 0.65,              # Technical reports
    "article": 0.60,             # Articles/blog posts
    "default": 0.65              # Fallback threshold
}

def get_similarity_threshold(doc_type: Optional[str] = None) -> float:
    """
    Get the similarity threshold for a given document type.
    
    Args:
        doc_type: Type of document (research_paper, thesis, assignment, etc.)
        
    Returns:
        float: Similarity threshold (0.0 - 1.0)
    """
    if doc_type and doc_type.lower() in SIMILARITY_THRESHOLDS:
        threshold = SIMILARITY_THRESHOLDS[doc_type.lower()]
        logger.info(f"Using threshold {threshold} for document type: {doc_type}")
        return threshold
    logger.info(f"Using default threshold {SIMILARITY_THRESHOLDS['default']}")
    return SIMILARITY_THRESHOLDS["default"]


# ---------------------------------------------------------------------------
# KEYWORD EXTRACTION FOR BETTER SEARCH
# ---------------------------------------------------------------------------
# Common academic stopwords to filter out
ACADEMIC_STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
    'this', 'that', 'these', 'those', 'it', 'its', 'we', 'our', 'they',
    'their', 'he', 'she', 'him', 'her', 'i', 'me', 'my', 'you', 'your',
    'which', 'who', 'whom', 'what', 'where', 'when', 'why', 'how',
    'paper', 'study', 'research', 'method', 'result', 'conclusion',
    'introduction', 'abstract', 'figure', 'table', 'section', 'chapter',
    'however', 'therefore', 'thus', 'hence', 'also', 'well', 'such',
    'used', 'using', 'based', 'proposed', 'presented', 'shown', 'found'
}


def extract_keywords_from_text(text: str, top_n: int = 10) -> List[str]:
    """
    Extract meaningful keywords from text using TF-based extraction.
    
    Args:
        text: The text to extract keywords from
        top_n: Number of top keywords to return
        
    Returns:
        List of keywords
    """
    if not text:
        return []
    
    # Clean and tokenize
    text = text.lower()
    # Remove special characters, keep alphanumeric and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    words = text.split()
    
    # Filter: remove stopwords, short words, and numbers
    filtered_words = [
        word for word in words 
        if word not in ACADEMIC_STOPWORDS 
        and len(word) > 3 
        and not word.isdigit()
    ]
    
    # Count word frequencies
    word_counts = Counter(filtered_words)
    
    # Get top keywords
    keywords = [word for word, _ in word_counts.most_common(top_n)]
    
    logger.debug(f"Extracted keywords: {keywords}")
    return keywords


def extract_abstract_from_markdown(md_content: str) -> str:
    """
    Extract abstract section from markdown content.
    
    Args:
        md_content: Full markdown content
        
    Returns:
        Abstract text or first 500 words if no abstract found
    """
    # Try to find abstract section
    abstract_patterns = [
        r'(?i)#+\s*abstract\s*\n(.*?)(?=\n#+|\Z)',
        r'(?i)abstract[:\s]*(.*?)(?=\n#+|introduction|\Z)',
    ]
    
    for pattern in abstract_patterns:
        match = re.search(pattern, md_content, re.DOTALL)
        if match:
            abstract = match.group(1).strip()
            if len(abstract) > 50:  # Valid abstract
                logger.info(f"Extracted abstract: {len(abstract)} characters")
                return abstract
    
    # Fallback: use first 500 words (skip title)
    lines = md_content.split('\n')
    content_lines = [l for l in lines[1:] if l.strip() and not l.startswith('#')]
    content = ' '.join(content_lines)
    words = content.split()[:500]
    fallback = ' '.join(words)
    logger.info(f"Using fallback content for keyword extraction: {len(fallback)} characters")
    return fallback


def build_search_query(md_content: str, title: str = "") -> str:
    """
    Build an optimized search query from document content.
    
    Combines title keywords with abstract/content keywords for better search results.
    
    Args:
        md_content: Full markdown content
        title: Document title
        
    Returns:
        Optimized search query string
    """
    # Extract abstract
    abstract = extract_abstract_from_markdown(md_content)
    
    # Extract keywords from abstract (priority)
    abstract_keywords = extract_keywords_from_text(abstract, top_n=8)
    
    # Extract keywords from title
    title_keywords = extract_keywords_from_text(title, top_n=4)
    
    # Combine: title keywords first, then abstract keywords (deduplicated)
    all_keywords = []
    for kw in title_keywords + abstract_keywords:
        if kw not in all_keywords:
            all_keywords.append(kw)
    
    # Take top 10 unique keywords
    final_keywords = all_keywords[:10]
    
    query = ' '.join(final_keywords)
    logger.info(f"Built search query: '{query}'")
    return query


# ---------------------------------------------------------------------------
# AUTHENTICATION
# ---------------------------------------------------------------------------
async def verify_token(request: Request):
    token = request.cookies.get("plagiarism-access-token")
    if token:
        token = token.replace("Bearer ", "")
        try:
            payload = jwt.decode(token, config["SECRET_KEY"], algorithms=[
                                 config["ALGORITHM"]])
        except ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return payload
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------------------
# FILE UTILITIES
# ---------------------------------------------------------------------------
def read_md_file(file_path):
    text = ""
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    return text


async def convert_pdf_to_md(file_path: str) -> str:
    output_folder = os.path.dirname(file_path)

    try:
        result = subprocess.run(
            ['marker_single', file_path, output_folder],
            text=True,
            check=True,
            stdout=subprocess.PIPE,
            shell=False
        )

    except subprocess.CalledProcessError as e:
        logger.error(f"PDF conversion error: {e.stderr}")
        return ""

    pdf_folder = os.path.splitext(os.path.basename(file_path))[0]
    pdf_output_dir = os.path.join(output_folder, pdf_folder)

    if not os.path.exists(pdf_output_dir):
        raise FileNotFoundError(f"Output folder {pdf_output_dir} not found.")

    # Get the markdown file in that folder
    md_file = [f for f in os.listdir(pdf_output_dir) if f.endswith(".md")]

    if not md_file:
        raise FileNotFoundError(
            "Markdown file not found in the output directory.")

    md_file_path = os.path.join(pdf_output_dir, md_file[0])
    return md_file_path


async def convert_docx_to_md(file_path: str) -> str:
    doc = DocxDocument(file_path)
    markdown_content = "\n".join([para.text for para in doc.paragraphs])
    return markdown_content


async def convert_to_md(file_path: str) -> str:
    """
    Convert a file (PDF or DOCX) to Markdown.
    """
    if file_path.endswith('.pdf'):
        return await convert_pdf_to_md(file_path)
    elif file_path.endswith('.docx'):
        return await convert_docx_to_md(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")


# ---------------------------------------------------------------------------
# SIMILARITY DETECTION
# ---------------------------------------------------------------------------
def detect_similarity(path1: str, path2: str, paper: dict, threshold: Optional[float] = None):
    """
    Detect similarity between two documents.
    
    Args:
        path1: Path to uploaded document markdown
        path2: Path to scraped paper markdown
        paper: Paper metadata dict
        threshold: Optional custom similarity threshold
    """
    paper_name = paper["title"]
    logger.info(f"Detecting similarity between uploaded paper and {paper_name}")

    # Use provided threshold or default
    sim_threshold = threshold if threshold is not None else SIMILARITY_THRESHOLDS["default"]

    # result for uploaded paper vs ith webscraped paper
    result = research_similarity.research_similarity(path1, path2, sim_threshold)
    
    if result is None:
        logger.warning(f"Similarity detection returned None for {paper_name}")
        return {
            "source": {"name": paper_name, "url": ""},
            "bert_score": 0.0,
            "tfidf_score": 0.0,
            "score": 0.0,
            "plagiarized_content": {"sources": []}
        }
    
    return {
        "source": {
            "name": paper_name,
            "url": "https://arxiv.org/abs/" + paper["doi"].split("arXiv.")[-1]
        },
        "bert_score": float(result["bert_score"]),
        "tfidf_score": float(result["tfidf_score"]),
        "score": float(result["score"]),
        "plagiarized_content": {
            "sources": result["plagiarized_content"]["sources"]
        }
    }


# ---------------------------------------------------------------------------
# AI DETECTION
# ---------------------------------------------------------------------------
def detect_ai_generated_content(file_path) -> List[AIGeneratedContent]:
    logger.info(f"Detecting AI Generated Content")
    roberta_score = roberta_ai_detection(file_path)
    detect_gpt_score = detect_gpt_main(file_path)

    return [
        AIGeneratedContent(method_name="Roberta Base Model",
                           score=roberta_score),
        AIGeneratedContent(method_name="Detect GPT", score=detect_gpt_score)
    ]


# ---------------------------------------------------------------------------
# PAPER SCRAPING (IMPROVED)
# ---------------------------------------------------------------------------
async def scrape_and_save_research_papers(
    query: str, 
    max_results: int = 5
) -> List[dict]:
    """
    Scrape research papers from arXiv using a search query.
    
    Args:
        query: Search query (keywords or title)
        max_results: Maximum number of papers to fetch (default: 5)
        
    Returns:
        List of paper details with paths
    """
    logger.info(f"Scraping papers from ArXiv with query: '{query[:50]}...'")

    output_folder = Path("scraped_papers")
    output_folder.mkdir(parents=True, exist_ok=True)

    result = get_arxiv_papers(query=query, max_results=max_results)

    scraped_papers = []

    for index, row in result.iterrows():
        doi = row.get("doi")
        title = row.get("title")
        journal = row.get("journal")

        if not doi:
            logger.debug(f"Skipping paper at index {index}: DOI not found.")
            continue

        filename = doi.replace("/", "_") + ".pdf"
        filepath = output_folder / filename

        try:
            save_pdf({"doi": doi}, filepath=str(filepath))
        except Exception as e:
            logger.warning(f"Failed to download paper {title}: {e}")
            continue

        paper_details = {
            "doi": doi,
            "title": title,
            "journal": journal,
            "path": str(filepath),
        }
        scraped_papers.append(paper_details)

    logger.info(f"Scraped {len(scraped_papers)} papers into '{output_folder}'")
    return scraped_papers

