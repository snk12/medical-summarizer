import time
import json
from datetime import datetime
from typing import Dict, Any, List
import torch
from transformers import pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import streamlit as st

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)

class ModelManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        self.load_all_models()

    @st.cache_resource
    def load_all_models(_self):
        models = {}
        
        with st.spinner("Loading Sentence Transformer..."):
            models['sentence_transformer'] = SentenceTransformer('all-MiniLM-L6-v2')
        
        with st.spinner("Loading BART Summarizer..."):
            try:
                models['bart_summarizer'] = hf_pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    device=-1,  # Force CPU
                    max_length=150,
                    min_length=50,
                    do_sample=False
                )
            except Exception as e:
                st.warning(f"BART model failed to load: {e}")
                models['bart_summarizer'] = None
        
        with st.spinner("Loading Medical NER..."):
            models['medical_ner'] = hf_pipeline(
                "ner",
                model="emilyalsentzer/Bio_ClinicalBERT",
                device=0 if _self.device == "cuda" else -1,
                aggregation_strategy="simple"
            )
        
        return models

    def get_sentence_embeddings(self, sentences):
        return self.models['sentence_transformer'].encode(sentences)

    def summarize_text(self, text, max_length=150, min_length=50):
        try:
            if 'bart_summarizer' not in self.models or self.models['bart_summarizer'] is None:
                raise Exception("BART model not available")
            
            words = text.split()
            if len(words) > 800:
                text = ' '.join(words[:800])
            
            result = self.models['bart_summarizer'](
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            return result[0]['summary_text']
        except Exception as e:
            raise Exception(f"BART summarization failed: {str(e)}")
    def extract_medical_entities(self, text):
        try:
            if 'medical_ner' not in self.models:
                return []
            
            entities = self.models['medical_ner'](text)
            processed_entities = []
            for entity in entities:
                processed_entities.append({
                    'text': entity.get('word', '').replace('##', ''),
                    'label': entity.get('entity_group', 'UNKNOWN'),
                    'confidence': entity.get('score', 0.5),
                    'start': entity.get('start', 0),
                    'end': entity.get('end', 0)
                })
            return processed_entities
        except Exception as e:
            # Silent fallback - don't show error to users
            return []

class MedicalTextProcessor:
    def __init__(self, model_manager):
        self.model_manager = model_manager
        download_nltk_data()
        self.stop_words = set(stopwords.words('english'))
        
        self.medical_terms = {
            'conditions': [
                'diabetes', 'hypertension', 'pneumonia', 'copd', 'asthma',
                'heart failure', 'myocardial infarction', 'stroke', 'sepsis',
                'kidney disease', 'liver disease', 'cancer', 'infection'
            ],
            'medications': [
                'insulin', 'metformin', 'lisinopril', 'atorvastatin', 'aspirin',
                'warfarin', 'furosemide', 'prednisone', 'antibiotics'
            ],
            'procedures': [
                'surgery', 'biopsy', 'catheterization', 'intubation',
                'dialysis', 'chemotherapy', 'radiation', 'transfusion'
            ]
        }
        
        self.medical_abbreviations = {
            'bp': 'blood pressure', 'hr': 'heart rate', 'rr': 'respiratory rate',
            'temp': 'temperature', 'o2': 'oxygen', 'iv': 'intravenous',
            'po': 'by mouth', 'bid': 'twice daily', 'tid': 'three times daily',
            'ekg': 'electrocardiogram', 'cxr': 'chest x-ray'
        }

    def preprocess_text(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""

        text = text.lower()
        
        # Expand medical abbreviations
        for abbrev, expansion in self.medical_abbreviations.items():
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
        
        # Clean text
        text = re.sub(r'[^\w\s\.\,\:\;\-\(\)\/]', ' ', text)
        text = re.sub(r'\b\d{1,2}\/\d{1,2}\/\d{2,4}\b', 'DATE', text)
        text = re.sub(r'\b\d{1,3}\/\d{1,3}\b', 'FRACTION', text)
        text = re.sub(r'\b\d+\.\d+\b', 'NUMBER', text)
        text = re.sub(r'\b\d+\b', 'NUMBER', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def extract_sentences(self, text: str) -> List[str]:
        try:
            sentences = sent_tokenize(text)
            return [s.strip() for s in sentences if len(s.strip()) > 10]
        except:
            sentences = text.split('.')
            return [s.strip() for s in sentences if len(s.strip()) > 10]

    def extract_medical_entities_enhanced(self, text: str) -> Dict:
        # Get AI entities
        ai_entities = self.model_manager.extract_medical_entities(text)
        
        # Get rule-based entities
        rule_entities = self.extract_entities_by_rules(text)
        
        # Combine results
        combined = {
            'conditions': [],
            'medications': [],
            'procedures': [],
            'general': []
        }
        
        # Add AI entities
        for entity in ai_entities:
            label = entity.get('label', 'UNKNOWN')
            category = self.categorize_ai_entity(label)
            combined[category].append({
                'text': entity.get('text', ''),
                'confidence': entity.get('confidence', 0.5),
                'method': 'ai',
                'label': label
            })
        
        # Add rule-based entities
        for category, entities in rule_entities.items():
            if category in combined:
                combined[category].extend(entities)
        
        # Remove duplicates
        for category in combined:
            seen = set()
            unique_entities = []
            for entity in combined[category]:
                text_key = entity['text'].lower()
                if text_key not in seen and text_key:
                    seen.add(text_key)
                    unique_entities.append(entity)
            combined[category] = unique_entities
        
        return combined

    def extract_entities_by_rules(self, text: str) -> Dict:
        entities = {
            'conditions': [],
            'medications': [],
            'procedures': []
        }
        
        text_lower = text.lower()
        for category, terms in self.medical_terms.items():
            for term in terms:
                if term in text_lower:
                    entities[category].append({
                        'text': term,
                        'confidence': 0.8,
                        'method': 'rule-based'
                    })
        
        return entities

    def categorize_ai_entity(self, label: str) -> str:
        label_mapping = {
            'PROBLEM': 'conditions',
            'TREATMENT': 'medications',
            'TEST': 'procedures',
            'PERSON': 'general',
            'ORG': 'general',
            'LOC': 'general'
        }
        return label_mapping.get(label, 'general')

    def analyze_text_structure(self, text: str) -> Dict:
        sentences = self.extract_sentences(text)
        words = word_tokenize(text.lower()) if text else []
        content_words = [w for w in words if w not in self.stop_words and w.isalpha()]
        
        return {
            'total_length': len(text),
            'sentence_count': len(sentences),
            'word_count': len(words),
            'content_word_count': len(content_words),
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'medical_term_density': self.calculate_medical_density(content_words)
        }

    def calculate_medical_density(self, words: List[str]) -> float:
        if not words:
            return 0.0
        
        medical_word_count = 0
        all_medical_terms = []
        
        for category_terms in self.medical_terms.values():
            all_medical_terms.extend(category_terms)
        
        for word in words:
            if any(term in word for term in all_medical_terms):
                medical_word_count += 1
        
        return medical_word_count / len(words)

class MedicalSummarizer:
    def __init__(self, model_manager, text_processor):
        self.model_manager = model_manager
        self.text_processor = text_processor
        self.extractive_sentence_count = 3

    def dual_summarize(self, text: str) -> Dict[str, str]:
        processed_text = self.text_processor.preprocess_text(text)
        
        extractive_summary = self.extractive_summarize(processed_text)
        abstractive_summary = self.abstractive_summarize(text)
        
        return {
            'extractive': extractive_summary,
            'abstractive': abstractive_summary,
            'method_comparison': {
                'extractive_length': len(extractive_summary.split()),
                'abstractive_length': len(abstractive_summary.split()),
                'length_ratio': len(abstractive_summary.split()) / max(1, len(extractive_summary.split()))
            }
        }

    def extractive_summarize(self, text: str) -> str:
        sentences = self.text_processor.extract_sentences(text)
        
        if len(sentences) <= self.extractive_sentence_count:
            return '. '.join(sentences)
        
        try:
            embeddings = self.model_manager.get_sentence_embeddings(sentences)
            
            # Calculate similarity to document centroid
            doc_centroid = np.mean(embeddings, axis=0).reshape(1, -1)
            similarities = cosine_similarity(embeddings, doc_centroid).flatten()
            
            # Add medical importance scores
            medical_scores = self.calculate_medical_importance_scores(sentences)
            
            # Combine scores
            final_scores = similarities * 0.6 + medical_scores * 0.4
            
            # Select top sentences
            top_indices = np.argsort(final_scores)[-self.extractive_sentence_count:]
            top_indices = sorted(top_indices)
            
            summary_sentences = [sentences[i] for i in top_indices]
            return '. '.join(summary_sentences)
            
        except Exception as e:
            # Fallback to TF-IDF
            return self.extractive_summarize_tfidf(sentences)

    def calculate_medical_importance_scores(self, sentences: List[str]) -> np.ndarray:
        scores = np.zeros(len(sentences))
        
        medical_importance_weights = {
            'conditions': 3.0,
            'medications': 2.5,
            'procedures': 2.0
        }
        
        for i, sentence in enumerate(sentences):
            entities = self.text_processor.extract_medical_entities_enhanced(sentence)
            
            medical_score = 0.0
            for category, entity_list in entities.items():
                if category in medical_importance_weights:
                    weight = medical_importance_weights[category]
                    for entity in entity_list:
                        confidence = entity.get('confidence', 0.5)
                        medical_score += weight * confidence
            
            scores[i] = medical_score
        
        # Normalize scores
        if scores.max() > 0:
            scores = scores / scores.max()
        
        return scores

    def extractive_summarize_tfidf(self, sentences: List[str]) -> str:
        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
            tfidf_matrix = vectorizer.fit_transform(sentences)
            sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
            
            top_indices = np.argsort(sentence_scores)[-self.extractive_sentence_count:]
            top_indices = sorted(top_indices)
            
            summary_sentences = [sentences[i] for i in top_indices]
            return '. '.join(summary_sentences)
            
        except Exception as e:
            return '. '.join(sentences[:self.extractive_sentence_count])

    def abstractive_summarize(self, text: str) -> str:
        try:
            if 'bart_summarizer' not in self.model_manager.models:
                return "Abstractive summary unavailable - using extractive method"
            
            summary = self.model_manager.summarize_text(text)
            return self.post_process_summary(summary)
        except Exception as e:
            return "Summary generation failed - model loading issue"

    def post_process_summary(self, summary: str) -> str:
        # Clean up summary
        summary = re.sub(r'\b(the patient|patient)\b', 'Patient', summary, flags=re.IGNORECASE)
        
        if not summary.endswith('.'):
            summary += '.'
        
        summary = re.sub(r'\s+', ' ', summary).strip()
        return summary

class MedicalRiskScorer:
    def __init__(self, text_processor):
        self.text_processor = text_processor
        
        self.severity_weights = {
            'myocardial infarction': 95, 'heart attack': 95, 'cardiac arrest': 100,
            'stroke': 90, 'sepsis': 95, 'pulmonary embolism': 90, 'heart failure': 80,
            'pneumonia': 70, 'copd': 65, 'diabetes': 60, 'kidney failure': 75,
            'liver failure': 80, 'cancer': 75, 'atrial fibrillation': 65,
            'hypertension': 50, 'asthma': 45, 'arthritis': 30, 'depression': 40,
            'anxiety': 35, 'migraine': 25, 'headache': 15, 'back pain': 25
        }
        
        self.medication_risks = {
            'warfarin': {'risk_level': 80, 'monitoring': 'required'},
            'insulin': {'risk_level': 70, 'monitoring': 'required'},
            'digoxin': {'risk_level': 75, 'monitoring': 'required'},
            'lithium': {'risk_level': 85, 'monitoring': 'required'},
            'methotrexate': {'risk_level': 80, 'monitoring': 'required'},
            'prednisone': {'risk_level': 60, 'monitoring': 'recommended'},
            'furosemide': {'risk_level': 50, 'monitoring': 'recommended'},
            'metformin': {'risk_level': 40, 'monitoring': 'routine'},
            'lisinopril': {'risk_level': 45, 'monitoring': 'routine'},
            'aspirin': {'risk_level': 30, 'monitoring': 'minimal'}
        }
        
        self.urgency_keywords = {
            'emergency': 100, 'urgent': 95, 'critical': 100, 'life-threatening': 100,
            'immediate': 95, 'stat': 100, 'severe': 80, 'acute': 75,
            'worsening': 70, 'deteriorating': 75, 'unstable': 80,
            'stable': 20, 'improving': 15, 'routine': 10
        }

    def calculate_comprehensive_risk_score(self, text: str, entities: Dict = None) -> Dict[str, float]:
        if entities is None:
            entities = self.text_processor.extract_medical_entities_enhanced(text)
        
        condition_risk = self.score_medical_conditions(entities, text)
        medication_risk = self.score_medications(entities, text)
        urgency_risk = self.score_urgency_indicators(text)
        interaction_risk = self.score_potential_interactions(entities)
        demographic_risk = self.score_demographic_factors(text)
        completeness_score = self.score_document_completeness(text)
        
        # Calculate overall risk (weighted combination)
        overall_risk = (
            condition_risk * 0.30 +
            medication_risk * 0.20 +
            urgency_risk * 0.20 +
            interaction_risk * 0.15 +
            demographic_risk * 0.10 +
            (100 - completeness_score) * 0.05
        )
        
        overall_risk = max(0, min(100, overall_risk))
        
        return {
            'overall_risk': overall_risk,
            'condition_severity': condition_risk,
            'medication_risk': medication_risk,
            'urgency_level': urgency_risk,
            'interaction_risk': interaction_risk,
            'demographic_risk': demographic_risk,
            'completeness_score': completeness_score,
            'risk_category': self.categorize_risk_level(overall_risk),
            'confidence_score': self.calculate_confidence_score(entities, text)
        }

    def score_medical_conditions(self, entities: Dict, text: str) -> float:
        condition_score = 0.0
        
        if 'conditions' in entities:
            for condition in entities['conditions']:
                condition_text = condition['text'].lower()
                severity = self.get_condition_severity(condition_text)
                confidence = condition.get('confidence', 0.5)
                weighted_severity = severity * confidence
                condition_score = max(condition_score, weighted_severity)
        
        # Check for critical conditions in text directly
        text_lower = text.lower()
        critical_conditions = [
            'cardiac arrest', 'myocardial infarction', 'heart attack',
            'stroke', 'sepsis', 'pulmonary embolism'
        ]
        
        for critical_condition in critical_conditions:
            if critical_condition in text_lower:
                condition_score = max(condition_score, 90)
        
        return min(100, condition_score)

    def get_condition_severity(self, condition: str) -> float:
        condition = condition.lower().strip()
        
        if condition in self.severity_weights:
            return self.severity_weights[condition]
        
        # Partial match
        for known_condition, severity in self.severity_weights.items():
            if known_condition in condition or condition in known_condition:
                return severity
        
        return 50.0  # Default severity

    def score_medications(self, entities: Dict, text: str) -> float:
        medication_score = 0.0
        high_risk_meds = 0
        
        if 'medications' in entities:
            for medication in entities['medications']:
                med_text = medication['text'].lower()
                
                if med_text in self.medication_risks:
                    med_risk = self.medication_risks[med_text]['risk_level']
                    confidence = medication.get('confidence', 0.5)
                    weighted_risk = med_risk * confidence
                    medication_score = max(medication_score, weighted_risk)
                    
                    if med_risk >= 70:
                        high_risk_meds += 1
        
        # Increase score for multiple high-risk medications
        if high_risk_meds > 1:
            medication_score *= (1 + (high_risk_meds - 1) * 0.2)
        
        return min(100, medication_score)

    def score_urgency_indicators(self, text: str) -> float:
        urgency_score = 0.0
        text_lower = text.lower()
        
        for keyword, weight in self.urgency_keywords.items():
            if keyword in text_lower:
                urgency_score = max(urgency_score, weight)
        
        # Check for vital sign abnormalities
        bp_matches = re.findall(r'\b(\d{2,3})/(\d{2,3})\b', text)
        for systolic, diastolic in bp_matches:
            systolic, diastolic = int(systolic), int(diastolic)
            if systolic > 180 or diastolic > 110:
                urgency_score = max(urgency_score, 90)
            elif systolic > 160 or diastolic > 100:
                urgency_score = max(urgency_score, 70)
        
        return urgency_score

    def score_potential_interactions(self, entities: Dict) -> float:
        interaction_score = 0.0
        
        medications = []
        if 'medications' in entities:
            medications = [med['text'].lower() for med in entities['medications']]
        
        # High-risk combinations
        high_risk_combinations = [
            (['warfarin', 'aspirin'], 70),
            (['insulin', 'alcohol'], 80),
            (['digoxin', 'furosemide'], 60)
        ]
        
        for combination, risk_score in high_risk_combinations:
            if all(any(item in med for med in medications) for item in combination):
                interaction_score = max(interaction_score, risk_score)
        
        # Multiple medications increase risk
        if len(medications) > 5:
            interaction_score = max(interaction_score, 40)
        elif len(medications) > 10:
            interaction_score = max(interaction_score, 60)
        
        return interaction_score

    def score_demographic_factors(self, text: str) -> float:
        demographic_score = 0.0
        
        # Extract age
        age_matches = re.findall(r'age[:\s]*(\d{1,3})', text.lower())
        if age_matches:
            age = int(age_matches[0])
            if age >= 80:
                demographic_score = 60
            elif age >= 65:
                demographic_score = 40
            elif age >= 50:
                demographic_score = 20
            elif age < 18:
                demographic_score = 30
        
        return demographic_score

    def score_document_completeness(self, text: str) -> float:
        text_lower = text.lower()
        requirements = [
            'age', 'gender', 'chief complaint', 'history', 'medications',
            'assessment', 'diagnosis', 'plan', 'follow-up'
        ]
        
        present_sections = sum(1 for req in requirements if req in text_lower)
        return (present_sections / len(requirements)) * 100

    def categorize_risk_level(self, overall_risk: float) -> str:
        if overall_risk >= 80:
            return "Critical"
        elif overall_risk >= 60:
            return "High"
        elif overall_risk >= 40:
            return "Moderate"
        elif overall_risk >= 20:
            return "Low"
        else:
            return "Minimal"

    def calculate_confidence_score(self, entities: Dict, text: str) -> float:
        total_entities = sum(len(entity_list) for entity_list in entities.values())
        if total_entities == 0:
            return 50.0
        
        total_confidence = 0
        for entity_list in entities.values():
            for entity in entity_list:
                total_confidence += entity.get('confidence', 0.5)
        
        avg_confidence = total_confidence / total_entities
        
        # Factor in document completeness
        completeness = self.score_document_completeness(text) / 100
        
        # Factor in text length
        length_confidence = min(1.0, len(text.split()) / 200)
        
        overall_confidence = (avg_confidence + completeness + length_confidence) / 3
        return overall_confidence * 100

class MedicalReportPipeline:
    def __init__(self):
        self.model_manager = ModelManager()
        self.text_processor = MedicalTextProcessor(self.model_manager)
        self.summarizer = MedicalSummarizer(self.model_manager, self.text_processor)
        self.risk_scorer = MedicalRiskScorer(self.text_processor)
        
        self.processing_stats = {
            'total_documents_processed': 0,
            'average_processing_time': 0.0,
            'success_rate': 0.0,
            'last_updated': datetime.now().isoformat()
        }

    def process_document(self, text: str, document_id: str = None) -> Dict[str, Any]:
        start_time = time.time()
        
        if document_id is None:
            document_id = f"doc_{int(time.time())}"
        
        try:
            # Text processing and entity extraction
            processed_text = self.text_processor.preprocess_text(text)
            entities = self.text_processor.extract_medical_entities_enhanced(processed_text)
            text_analysis = self.text_processor.analyze_text_structure(processed_text)
            
            # Summarization
            summaries = self.summarizer.dual_summarize(text)
            
            # Risk assessment
            risk_scores = self.risk_scorer.calculate_comprehensive_risk_score(processed_text, entities)
            
            processing_time = time.time() - start_time
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(text, summaries, entities, risk_scores)
            
            result = {
                'document_info': {
                    'document_id': document_id,
                    'processed_at': datetime.now().isoformat(),
                    'processing_time_seconds': processing_time,
                    'original_length': len(text),
                    'processed_length': len(processed_text)
                },
                'text_analysis': text_analysis,
                'entities': entities,
                'summaries': summaries,
                'risk_assessment': {
                    'scores': risk_scores
                },
                'quality_metrics': quality_metrics,
                'processing_metadata': {
                    'pipeline_version': '1.0',
                    'success': True,
                    'error_message': None
                }
            }
            
            self._update_processing_stats(processing_time, success=True)
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_processing_stats(processing_time, success=False)
            
            return {
                'document_info': {
                    'document_id': document_id,
                    'processed_at': datetime.now().isoformat(),
                    'processing_time_seconds': processing_time,
                    'original_length': len(text)
                },
                'processing_metadata': {
                    'pipeline_version': '1.0',
                    'success': False,
                    'error_message': str(e)
                }
            }

    def _calculate_quality_metrics(self, original_text: str, summaries: Dict, entities: Dict, risk_scores: Dict) -> Dict:
        # Calculate entity extraction quality
        total_entities = sum(len(entity_list) for entity_list in entities.values())
        avg_entity_confidence = 0
        if total_entities > 0:
            total_confidence = sum(
                sum(entity.get('confidence', 0.5) for entity in entity_list)
                for entity_list in entities.values()
            )
            avg_entity_confidence = total_confidence / total_entities
        
        # Calculate summary quality
        extractive_compression = len(summaries['extractive']) / len(original_text) if original_text else 0
        abstractive_compression = len(summaries['abstractive']) / len(original_text) if original_text else 0
        
        # Overall quality score
        quality_components = [
            avg_entity_confidence,
            min(1.0, risk_scores['confidence_score'] / 100),
            1.0 - abs(0.3 - extractive_compression),
            1.0 - abs(0.2 - abstractive_compression)
        ]
        
        overall_quality = sum(quality_components) / len(quality_components)
        
        return {
            'overall_quality_score': overall_quality,
            'entity_extraction_quality': avg_entity_confidence,
            'total_entities_found': total_entities,
            'extractive_compression_ratio': extractive_compression,
            'abstractive_compression_ratio': abstractive_compression,
            'risk_assessment_confidence': risk_scores['confidence_score'],
            'processing_completeness': 1.0
        }

    def _update_processing_stats(self, processing_time: float, success: bool):
        self.processing_stats['total_documents_processed'] += 1
        
        # Update average processing time
        prev_avg = self.processing_stats['average_processing_time']
        n = self.processing_stats['total_documents_processed']
        self.processing_stats['average_processing_time'] = (prev_avg * (n-1) + processing_time) / n
        
        # Update success rate
        if success:
            prev_success_count = self.processing_stats['success_rate'] * (n-1)
            self.processing_stats['success_rate'] = (prev_success_count + 1) / n
        else:
            prev_success_count = self.processing_stats['success_rate'] * (n-1)
            self.processing_stats['success_rate'] = prev_success_count / n
        
        self.processing_stats['last_updated'] = datetime.now().isoformat()

    def get_processing_stats(self) -> Dict:
        return self.processing_stats.copy()
