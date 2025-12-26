import re
import logging
import nltk # type: ignore
from nltk.stem import WordNetLemmatizer # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
from sentence_transformers import SentenceTransformer # type: ignore
import numpy as np
import faiss # type: ignore
from transformers import AutoTokenizer, AutoModel #type: ignore
from typing import List, Tuple
import torch # type: ignore
from nltk.corpus import stopwords # type: ignore
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger("LY-PROJECT")


class GetPapers:
    def load_papers(self, paper1_path, paper2_path):

        try:
            with open(paper1_path, "r", encoding='utf-8') as f1:
                paper1_content = f1.read()
            with open(paper2_path, "r", encoding='utf-8') as f2:
                paper2_content = f2.read()

            return paper1_content, paper2_content

        except Exception as e:
            logger.error(f'Error reading files: {e}')
            return None, None


class Preprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.word_lemmatizer = WordNetLemmatizer()
        
        self.cleaning_pattern = re.compile(r'[^a-zA-Z0-9\s.,]')

        
        # Pre-compile the regex patterns for section extraction
        self.section_patterns = {
            'abstract': re.compile(
                r'(?i)(?:^|\n)#{1,6}\s*(?:\d+\s+)?(?:.*?abstract|.*?summary).*?\n(.*?)(?=\n#{1,6}|$)',
                re.DOTALL),
            'introduction': re.compile(
                r'(?i)(?:^|\n)#{1,6}\s*(?:\d+\s+)?(?:.*?introduction|.*?background).*?\n(.*?)(?=\n#{1,6}|$)',
                re.DOTALL),
            'methodology': re.compile(
                r'(?i)(?:^|\n)#{1,6}\s*(?:\d+\s+)?(?:.*?methodology|.*?methods).*?\n(.*?)(?=\n#{1,6}|$)',
                re.DOTALL),
            'results': re.compile(
                r'(?i)(?:^|\n)#{1,6}\s*(?:\d+\s+)?(?:.*?results|.*?findings).*?\n(.*?)(?=\n#{1,6}|$)',
                re.DOTALL),
            'discussion': re.compile(
                r'(?i)(?:^|\n)#{1,6}\s*(?:\d+\s+)?(?:.*?discussion|.*?interpretation).*?\n(.*?)(?=\n#{1,6}|$)',
                re.DOTALL),
            'conclusion': re.compile(
                r'(?i)(?:^|\n)#{1,6}\s*(?:\d+\s+)?(?:.*?conclusion.*?(?:future)?|.*?final).*?\n(.*?)(?=\n#{1,6}|$)',
                re.DOTALL)
        }

    def extract_sections(self, paper_content):
        sections = {
            'abstract': '',
            'introduction': '',
            'methodology': '',
            'results': '',
            'discussion': '',
            'conclusion': '',
            'full_text': paper_content,
        }

        for section, pattern in self.section_patterns.items():
            match = pattern.search(paper_content)
            if match:
                content = match.group(1).strip()
                if content:
                    logger.debug(f"Found {section.upper()}: {len(content)} chars")
                    sections[section] = content
                else:
                    logger.debug(f"Empty {section.upper()} section")
            else:
                logger.debug(f"No {section.upper()} section found")

        return sections   
    
    def preprocess_text(self, text, use_lemmatization=True):
        if not text:
            return ''
        try:
            # Use the precompiled pattern for cleaning text
            text = self.cleaning_pattern.sub(' ', text).lower()
            tokens = nltk.word_tokenize(text)
            processed_tokens = [
                self.word_lemmatizer.lemmatize(token) if use_lemmatization else token
                for token in tokens if token not in self.stop_words
            ]
            return ' '.join(processed_tokens)
        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            return ''


class SimilarityCalculator:
    def __init__(self, bert_model='all-MiniLM-L6-v2', max_chunk_length=510):
        self.tfidf_vectorizer = TfidfVectorizer()
        self.bert_model = SentenceTransformer(bert_model)
        self.max_chunk_length = max_chunk_length

    def _chunk_text(self, text):
        """Splits text into approximate word chunks before tokenizing."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.max_chunk_length):
            chunk = ' '.join(words[i:i + self.max_chunk_length])
            chunks.append(chunk)
        return chunks

    def calculate_tfidf_similarity(self, text1, text2):
        tfidf_matrix = self.tfidf_vectorizer.fit_transform([text1, text2])
        similarity_score = cosine_similarity(
            tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity_score

    def calculate_transformer_similarity(self, text1, text2):
        text1_chunks = self._chunk_text(text1)
        text2_chunks = self._chunk_text(text2)

        similarities = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_pair = {
                executor.submit(self.calculate_bert_similarity, chunk1, chunk2): (chunk1, chunk2)
                for chunk1 in text1_chunks for chunk2 in text2_chunks
            }
            for future in as_completed(future_to_pair):
                try:
                    sim = future.result()
                    similarities.append(sim)
                except Exception as e:
                    logger.error(f"Error in chunk similarity calculation: {e}")
                    similarities.append(0.0)

        avg_similarity = np.mean(similarities) if similarities else 0.0
        return avg_similarity

    def calculate_bert_similarity(self, text1, text2):
        try:
            embeddings = self.bert_model.encode([text1, text2])
            similarity_score = cosine_similarity(
                [embeddings[0]], [embeddings[1]])[0][0]
            return similarity_score
        except Exception as e:
            logger.error(f"Error in BERT similarity calculation for chunk: {e}")
            return 0.0

    def combined_similarity(self, text1, text2, tfidf_weight=0.3, bert_weight=0.7):
        tfidf_score = self.calculate_tfidf_similarity(text1, text2)
        bert_score = self.calculate_transformer_similarity(text1, text2)
        combined_score = (tfidf_weight * tfidf_score) + \
            (bert_weight * bert_score)
        return combined_score, {"TF-IDF": tfidf_score, "BERT": bert_score}


class PlagiarismDetector:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Pre-compile regex patterns for cleaning and sentence splitting
        self.whitespace_pattern = re.compile(r'\s+')
        self.sentence_split_pattern = re.compile(r'(?<=[.!?])\s+')
        self.word_pattern = re.compile(r'\b\w+\b')

    def _clean_text(self, text: str) -> str:
        text = self.whitespace_pattern.sub(' ', text)
        text = re.sub(r'\.(?=[A-Za-z])', '. ', text)
        return text.strip()
    
    def _split_into_sentences(self, text: str) -> List[str]:
        text = self._clean_text(text)
        potential_sentences = self.sentence_split_pattern.split(text)
        valid_sentences = []
        for sent in potential_sentences:
            sent = sent.strip()
            word_count = len(self.word_pattern.findall(sent))
            if word_count >= 4:
                valid_sentences.append(sent)
        return valid_sentences

    def _get_sentence_embeddings(self, sentences: List[str], batch_size=32) -> np.ndarray:
        """Generate embeddings for sentences using batching"""
        embeddings = []
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            
            # Tokenize with longer max length to handle longer sentences
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            with torch.no_grad():
                outputs = self.model(**encoded)
                # Use mean pooling instead of just CLS token
                attention_mask = encoded['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sentence_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embeddings.append(sentence_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)

    def find_plagiarized_pairs(self, 
                             source_text: str, 
                             target_text: str,
                             similarity_threshold: float = 0.65) -> List[Tuple[str, str, float]]:
        """Find similar sentence pairs with improved accuracy"""
        # Split into sentences with improved method
        source_sentences = self._split_into_sentences(source_text)
        target_sentences = self._split_into_sentences(target_text)
        
        if not source_sentences or not target_sentences:
            return []

        # Get embeddings
        source_embeddings = self._get_sentence_embeddings(source_sentences)
        target_embeddings = self._get_sentence_embeddings(target_sentences)
        
        # Normalize embeddings
        faiss.normalize_L2(source_embeddings)
        faiss.normalize_L2(target_embeddings)
        
        # Create FAISS index
        dimension = source_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(source_embeddings)
        
        # Find similar sentences
        similarities, indices = index.search(target_embeddings, k=1)
        
        # Collect valid matches
        plagiarized_pairs = []
        for i, (similarity, idx) in enumerate(zip(similarities, indices)):
            similarity_score = similarity[0]
            
            if similarity_score >= similarity_threshold:
                source_sent = source_sentences[idx[0]]
                target_sent = target_sentences[i]
                
                # Additional validation
                if self._is_valid_match(source_sent, target_sent):
                    plagiarized_pairs.append((source_sent, target_sent, similarity_score))
        
        return plagiarized_pairs
    
    def _is_valid_match(self, source_sent: str, target_sent: str) -> bool:
        """Additional validation for matched pairs"""
        # Ignore if either sentence is too short
        if len(source_sent.split()) < 4 or len(target_sent.split()) < 4:
            return False
            
        # Ignore if sentences are just numbers or punctuation
        if not re.search(r'[A-Za-z]', source_sent) or not re.search(r'[A-Za-z]', target_sent):
            return False
        
        # Ignore if sentences are too different in length
        source_words = len(source_sent.split())
        target_words = len(target_sent.split())
        if max(source_words, target_words) / min(source_words, target_words) > 2:
            return False
            
        return True

    def get_plagiarized_sentences(self, 
                                    source_text: str, 
                                    target_text: str,
                                    similarity_threshold: float = 0.65) -> dict:
            """
            Returns dictionary containing arrays of plagiarized sentences and their sources
            """
            pairs = self.find_plagiarized_pairs(source_text, target_text, similarity_threshold)
            
            return {
                'plagiarized_sentences': [pair[1] for pair in pairs],  # Sentences from paper2
                'source_sentences': [pair[0] for pair in pairs],       # Original sentences from paper1
                'similarity_scores': [pair[2] for pair in pairs]       # Similarity scores
            }


def research_similarity(path1, path2, similarity_threshold: float = 0.65):
    """
    Compute similarity between two documents.
    
    Args:
        path1: Path to first document (markdown)
        path2: Path to second document (markdown)
        similarity_threshold: Threshold for plagiarism detection (default: 0.65)
        
    Returns:
        dict with similarity scores and plagiarized content
    """
    get_papers = GetPapers()
    detector = PlagiarismDetector()
    preprocessor = Preprocessor()
    similarity_calculator = SimilarityCalculator()
    
    paper1, paper2 = get_papers.load_papers(path1, path2)

    if not paper1 or not paper2:
        logger.error("Error loading papers")
        return None
    
    plagiarism_results = detector.get_plagiarized_sentences(paper1, paper2, similarity_threshold)

    sections_paper1 = preprocessor.extract_sections(paper1)
    sections_paper2 = preprocessor.extract_sections(paper2)

    if not sections_paper1 or not sections_paper2:
        logger.error("Error extracting sections")
        return None


    logger.info('Computing similarity scores by sections')

    for section in sections_paper1.keys():
        text1 = preprocessor.preprocess_text(sections_paper1[section])
        text2 = preprocessor.preprocess_text(sections_paper2[section])

        if text1 and text2:
            combined_score, individual_scores = similarity_calculator.combined_similarity(
                text1, text2)
            tfidf_score = similarity_calculator.calculate_tfidf_similarity(
                text1, text2)
            bert_score = similarity_calculator.calculate_transformer_similarity(
                text1, text2)
            
            logger.debug(
                f"{section.capitalize():<15} : Combined Score: {combined_score:.4f}")
            logger.debug(f"Individual Scores: TF-IDF: {individual_scores['TF-IDF']:.4f}, "
                  f"BERT: {individual_scores['BERT']:.4f}")
        else:
            logger.debug(f"{section.capitalize():<15} : No content available")

    return {
            "data": {
                "name": "src1", 
                "url": "http://abc.com"
            },
            "plagiarized_content": {
                "sentences": plagiarism_results['plagiarized_sentences'],
                "sources": plagiarism_results['source_sentences'],
                "scores": plagiarism_results['similarity_scores']
            },
            "bert_score": bert_score,
            "tfidf_score": tfidf_score,
            "score": combined_score
        }