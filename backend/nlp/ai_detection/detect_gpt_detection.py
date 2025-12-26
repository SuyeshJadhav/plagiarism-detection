import logging
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from functools import lru_cache

logger = logging.getLogger("LY-PROJECT")

# Lazy loading for model
_model_cache = {}

def get_gpt2_model():
    """Lazy load GPT-2 model with caching."""
    if "gpt2" not in _model_cache:
        model_name = "gpt2"
        logger.info(f"Loading GPT-2 model for DetectGPT: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
        _model_cache["gpt2"] = (tokenizer, model)
    return _model_cache["gpt2"]


def compute_log_probability(tokenizer, model, text: str) -> float:
    """Calculate log probability of a text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        log_prob = -outputs.loss.item()
    return log_prob


def generate_perturbations(text: str, num_perturbations: int = 5) -> list:
    """Generate perturbed versions of the text using synonym replacement."""
    words = text.split()
    synonym_dict = {
        "good": ["great", "excellent", "nice"],
        "bad": ["terrible", "awful", "poor"],
        "quick": ["fast", "rapid", "speedy"],
        "slow": ["sluggish", "lethargic", "unhurried"],
        "important": ["crucial", "significant", "vital"],
        "large": ["big", "massive", "substantial"],
        "small": ["tiny", "little", "minute"],
    }

    perturbed_texts = []
    for _ in range(num_perturbations):
        perturbed_words = [
            random.choice(synonym_dict.get(word.lower(), [word])) 
            if random.random() < 0.3 else word
            for word in words
        ]
        perturbed_texts.append(" ".join(perturbed_words))
    return perturbed_texts


def detect_ai_generated_text(text: str, num_perturbations: int = 5, threshold: float = 0.5) -> tuple:
    """
    Detect AI-generated text using the DetectGPT algorithm.
    
    Uses perturbation-based detection: AI-generated text tends to have
    negative curvature in the log probability space.
    
    Args:
        text: Text to analyze
        num_perturbations: Number of perturbations to generate
        threshold: Curvature threshold for classification
        
    Returns:
        tuple: (classification, curvature_score)
    """
    tokenizer, model = get_gpt2_model()
    
    # Calculate log probability of the original text
    original_log_prob = compute_log_probability(tokenizer, model, text)

    # Generate perturbed texts and compute their log probabilities
    perturbed_texts = generate_perturbations(text, num_perturbations=num_perturbations)
    perturbed_log_probs = [
        compute_log_probability(tokenizer, model, perturbed) 
        for perturbed in perturbed_texts
    ]

    if not perturbed_log_probs:
        raise ValueError("No perturbed log probabilities available.")

    # Calculate average log probability of perturbed texts
    avg_perturbed_log_prob = sum(perturbed_log_probs) / len(perturbed_log_probs)

    # Calculate curvature score
    curvature_score = original_log_prob - avg_perturbed_log_prob

    # Classify based on curvature score
    if curvature_score > threshold:
        return "AI-generated", curvature_score
    else:
        return "Human-written", curvature_score


def detect_gpt_main(file_path: str) -> float:
    """
    Main function to detect AI-generated content using DetectGPT.
    
    Args:
        file_path: Path to the markdown file to analyze
        
    Returns:
        float: Curvature score (higher = more likely AI-generated)
    """
    from routers.utils import read_md_file
    
    try:
        text = read_md_file(file_path)
        if not text:
            raise ValueError("The input text file is empty or cannot be read.")
        
        # Truncate very long texts for performance
        words = text.split()
        if len(words) > 1000:
            text = ' '.join(words[:1000])
            logger.info("Text truncated to 1000 words for DetectGPT analysis")
        
        classification, curvature_score = detect_ai_generated_text(text)

        logger.info("DetectGPT Analysis Complete")
        logger.info(f"  Classification: {classification}")
        logger.info(f"  Curvature Score: {curvature_score:.4f}")
        
        return curvature_score
        
    except Exception as e:
        logger.error(f"DetectGPT Error: {e}")
        return None
