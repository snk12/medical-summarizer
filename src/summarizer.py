import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List, Tuple
import re

class MedicalSummarizer:
    def __init__(self, model_manager, text_processor):
        self.model_manager = model_manager
        self.text_processor = text_processor
        self.extractive_sentence_count = 3
        self.abstractive_max_length = 150
        self.abstractive_min_length = 50
        self.medical_importance_weights = {
            'conditions': 3.0, 'medications': 2.5, 'procedures': 2.0,
            'measurements': 1.5, 'anatomy': 1.0
        }
    
    def dual_summarize(self, text: str) -> Dict[str, str]:
        processed_text = self.text_processor.preprocess_text(text)
        extractive_summary = self.extractive_summarize(processed_text)
        abstractive_summary = self.abstractive_summarize(text)
        
        return {
            'extractive': extractive_summary,
            'abstractive': abstractive_summary,
            'method_comparison': self.compare_summaries(extractive_summary, abstractive_summary)
        }
    
    def extractive_summarize(self, text: str, num_sentences: int = None) -> str:
        if num_sentences is None:
            num_sentences = self.extractive_sentence_count
        
        sentences = self.text_processor.extract_sentences(text)
        if len(sentences) <= num_sentences:
            return '. '.join(sentences)
        
        try:
            embeddings = self.model_manager.get_sentence_embeddings(sentences)
            similarity_scores = self.calculate_similarity_scores(embeddings)
            medical_scores = self.calculate_medical_importance_scores(sentences)
            position_scores = self.calculate_position_scores(len(sentences))
            
            final_scores = (similarity_scores * 0.4 + medical_scores * 0.4 + position_scores * 0.2)
            top_indices = np.argsort(final_scores)[-num_sentences:]
            top_indices = sorted(top_indices)
            
            summary_sentences = [sentences[i] for i in top_indices]
            return '. '.join(summary_sentences)
        except Exception as e:
            return self.extractive_summarize_tfidf(sentences, num_sentences)
    
    def calculate_similarity_scores(self, embeddings: np.ndarray) -> np.ndarray:
        doc_centroid = np.mean(embeddings, axis=0).reshape(1, -1)
        similarities = cosine_similarity(embeddings, doc_centroid).flatten()
        return similarities
    
    def calculate_medical_importance_scores(self, sentences: List[str]) -> np.ndarray:
        scores = np.zeros(len(sentences))
        for i, sentence in enumerate(sentences):
            entities = self.text_processor.extract_medical_entities_enhanced(sentence)
            medical_score = 0.0
            for category, entity_list in entities.items():
                if category in self.medical_importance_weights:
                    weight = self.medical_importance_weights[category]
                    for entity in entity_list:
                        confidence = entity.get('confidence', 0.5)
                        medical_score += weight * confidence
            scores[i] = medical_score
        
        if scores.max() > 0:
            scores = scores / scores.max()
        return scores
    
    def calculate_position_scores(self, num_sentences: int) -> np.ndarray:
        scores = np.zeros(num_sentences)
        for i in range(num_sentences):
            if i == 0:
                scores[i] = 1.0
            elif i == num_sentences - 1:
                scores[i] = 0.8
            elif i < 3:
                scores[i] = 0.6
            else:
                scores[i] = 0.3
        return scores
    
    def extractive_summarize_tfidf(self, sentences: List[str], num_sentences: int) -> str:
        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
            tfidf_matrix = vectorizer.fit_transform(sentences)
            sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
            top_indices = np.argsort(sentence_scores)[-num_sentences:]
            top_indices = sorted(top_indices)
            summary_sentences = [sentences[i] for i in top_indices]
            return '. '.join(summary_sentences)
        except Exception as e:
            return '. '.join(sentences[:num_sentences])
    
    def abstractive_summarize(self, text: str, max_length: int = None, min_length: int = None) -> str:
        if max_length is None:
            max_length = self.abstractive_max_length
        if min_length is None:
            min_length = self.abstractive_min_length
        
        try:
            prepared_text = self.prepare_text_for_bart(text)
            summary = self.model_manager.summarize_text(prepared_text, max_length=max_length, min_length=min_length)
            return self.post_process_summary(summary)
        except Exception as e:
            return self.extractive_summarize(text, num_sentences=2)
    
    def prepare_text_for_bart(self, text: str) -> str:
        words = text.split()
        max_words = 800
        if len(words) > max_words:
            sentences = self.text_processor.extract_sentences(text)
            if len(sentences) > 1:
                important_content = self.extractive_summarize(text, num_sentences=min(5, len(sentences)))
                words = important_content.split()
                if len(words) > max_words:
                    words = words[:max_words]
        return ' '.join(words)
    
    def post_process_summary(self, summary: str) -> str:
        summary = re.sub(r'(the patient|patient)', 'Patient', summary, flags=re.IGNORECASE)
        if not summary.endswith('.'):
            summary += '.'
        summary = re.sub(r'\s+', ' ', summary).strip()
        return summary
    
    def compare_summaries(self, extractive: str, abstractive: str) -> Dict:
        return {
            'extractive_length': len(extractive.split()),
            'abstractive_length': len(abstractive.split()),
            'length_ratio': len(abstractive.split()) / max(1, len(extractive.split())),
            'overlap_score': self.calculate_overlap_score(extractive, abstractive)
        }
    
    def calculate_overlap_score(self, text1: str, text2: str) -> float:
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0.0
    
    def generate_summary_with_metadata(self, text: str) -> Dict:
        summaries = self.dual_summarize(text)
        entities = self.text_processor.extract_medical_entities_enhanced(text)
        text_analysis = self.text_processor.analyze_text_structure(text)
        
        return {
            'summaries': summaries,
            'entities': entities,
            'text_analysis': text_analysis,
            'processing_metadata': {
                'original_length': len(text),
                'extractive_compression_ratio': len(summaries['extractive']) / len(text),
                'abstractive_compression_ratio': len(summaries['abstractive']) / len(text)
            }
        }
