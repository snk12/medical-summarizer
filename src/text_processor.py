import re
import nltk
import pandas as pd
from typing import List, Dict, Tuple
import string
from collections import Counter

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

class MedicalTextProcessor:
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.stop_words = set(stopwords.words('english'))
        self.medical_stop_words = {
            'patient', 'pt', 'year', 'old', 'male', 'female', 'admission',
            'discharge', 'hospital', 'day', 'time', 'given', 'noted'
        }
        self.medical_terms = self.load_medical_terminology()
        self.medical_abbreviations = self.load_medical_abbreviations()
    
    def load_medical_terminology(self):
        return {
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
            ],
            'anatomy': [
                'heart', 'lung', 'kidney', 'liver', 'brain', 'chest',
                'abdomen', 'extremities', 'head', 'neck'
            ]
        }
    
    def load_medical_abbreviations(self):
        return {
            'bp': 'blood pressure', 'hr': 'heart rate', 'rr': 'respiratory rate',
            'temp': 'temperature', 'o2': 'oxygen', 'iv': 'intravenous',
            'po': 'by mouth', 'bid': 'twice daily', 'tid': 'three times daily',
            'qid': 'four times daily', 'prn': 'as needed', 'stat': 'immediately',
            'ekg': 'electrocardiogram', 'cxr': 'chest x-ray'
        }
    
    def preprocess_text(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = self.expand_abbreviations(text)
        text = self.clean_medical_text(text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def expand_abbreviations(self, text: str) -> str:
        for abbrev, expansion in self.medical_abbreviations.items():
            pattern = r'' + re.escape(abbrev) + r''
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
        return text
    
    def clean_medical_text(self, text: str) -> str:
        text = re.sub(r'[^\w\s\.\,\:\;\-\(\)\/]', ' ', text)
        text = re.sub(r'\d{1,2}\/\d{1,2}\/\d{2,4}', 'DATE', text)
        text = re.sub(r'\d{1,3}\/\d{1,3}', 'FRACTION', text)
        text = re.sub(r'\d+\.\d+', 'NUMBER', text)
        text = re.sub(r'\d+', 'NUMBER', text)
        return text
    
    def extract_sentences(self, text: str) -> List[str]:
        try:
            sentences = sent_tokenize(text)
            return [s.strip() for s in sentences if len(s.strip()) > 10]
        except:
            sentences = text.split('.')
            return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def extract_medical_entities_enhanced(self, text: str) -> Dict:
        ai_entities = self.model_manager.extract_medical_entities(text)
        rule_entities = self.extract_entities_by_rules(text)
        return self.combine_entity_results(ai_entities, rule_entities)
    
    def extract_entities_by_rules(self, text: str) -> Dict:
        entities = {
            'conditions': [], 'medications': [], 'procedures': [], 
            'anatomy': [], 'measurements': []
        }
        
        text_lower = text.lower()
        for category, terms in self.medical_terms.items():
            for term in terms:
                if term in text_lower:
                    entities[category].append({
                        'text': term, 'confidence': 0.8, 'method': 'rule-based'
                    })
        return entities
    
    def combine_entity_results(self, ai_entities: List, rule_entities: Dict) -> Dict:
        combined = {
            'conditions': [], 'medications': [], 'procedures': [],
            'anatomy': [], 'measurements': [], 'general': []
        }
        
        for entity in ai_entities:
            category = self.categorize_ai_entity(entity['label'])
            combined[category].append({
                'text': entity['text'], 'confidence': entity['confidence'],
                'method': 'ai', 'label': entity['label']
            })
        
        for category, entities in rule_entities.items():
            combined[category].extend(entities)
        
        return combined
    
    def categorize_ai_entity(self, label: str) -> str:
        label_mapping = {
            'PROBLEM': 'conditions', 'TREATMENT': 'medications',
            'TEST': 'procedures', 'PERSON': 'general', 'ORG': 'general'
        }
        return label_mapping.get(label, 'general')
    
    def analyze_text_structure(self, text: str) -> Dict:
        sentences = self.extract_sentences(text)
        words = word_tokenize(text.lower())
        content_words = [w for w in words if w not in self.stop_words]
        
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
