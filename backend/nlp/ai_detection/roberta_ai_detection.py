import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from functools import lru_cache

logger = logging.getLogger("LY-PROJECT")

# Lazy loading for models
_model_cache = {}

def get_roberta_model():
    """Lazy load RoBERTa model with caching."""
    if "roberta" not in _model_cache:
        model_name = "roberta-large-openai-detector"
        logger.info(f"Loading RoBERTa model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        _model_cache["roberta"] = (tokenizer, model)
    return _model_cache["roberta"]


def chunk_text(text: str, max_words: int = 400) -> list:
    """
    Split text into chunks of approximately max_words.
    RoBERTa has 512 token limit, ~400 words is safe.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = ' '.join(words[i:i + max_words])
        if chunk.strip():
            chunks.append(chunk)
    return chunks if chunks else [text]


def analyze_chunk(tokenizer, model, chunk: str) -> tuple:
    """Analyze a single chunk and return (prediction_index, confidence)."""
    inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).squeeze()
    
    prediction_idx = torch.argmax(probabilities).item()
    confidence = probabilities[1].item()  # Probability of AI-generated (index 1)
    
    return prediction_idx, confidence


def roberta_ai_detection(file_path: str) -> float:
    """
    Detect AI-generated content using RoBERTa model with chunked analysis.
    
    Splits the document into chunks to overcome the 512 token limit,
    analyzes each chunk, and returns a weighted average score.
    
    Args:
        file_path: Path to the markdown file to analyze
        
    Returns:
        float: AI-generated probability score (0.0 = human, 1.0 = AI)
    """
    from routers.utils import read_md_file
    
    text = read_md_file(file_path)
    if not text or not text.strip():
        logger.warning("Empty file provided for AI detection")
        return 0.0
    
    tokenizer, model = get_roberta_model()
    
    # Split into chunks
    chunks = chunk_text(text, max_words=400)
    logger.info(f"Analyzing {len(chunks)} chunks for AI detection")
    
    if len(chunks) == 0:
        return 0.0
    
    # Analyze each chunk
    chunk_scores = []
    for i, chunk in enumerate(chunks):
        try:
            _, confidence = analyze_chunk(tokenizer, model, chunk)
            chunk_scores.append(confidence)
            logger.debug(f"Chunk {i+1}/{len(chunks)}: AI probability = {confidence:.3f}")
        except Exception as e:
            logger.error(f"Error analyzing chunk {i+1}: {e}")
            continue
    
    if not chunk_scores:
        logger.error("No chunks could be analyzed")
        return 0.0
    
    # Calculate weighted average (give more weight to higher scores)
    # This helps catch AI-generated sections even if most content is human-written
    avg_score = sum(chunk_scores) / len(chunk_scores)
    max_score = max(chunk_scores)
    
    # Weighted combination: 70% average + 30% max
    # This balances overall detection with catching partial AI content
    final_score = 0.7 * avg_score + 0.3 * max_score
    
    labels = ["Human-written", "AI-generated"]
    prediction = labels[1] if final_score > 0.5 else labels[0]
    
    logger.info(f"RoBERTa AI Detection: {prediction} | Score: {final_score:.3f}")
    logger.info(f"  Chunks analyzed: {len(chunk_scores)}, Avg: {avg_score:.3f}, Max: {max_score:.3f}")
    
    return final_score
