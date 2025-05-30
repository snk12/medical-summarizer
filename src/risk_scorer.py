import re
import json
from typing import Dict, List, Tuple
import pandas as pd
from collections import Counter, defaultdict
import numpy as np

class MedicalRiskScorer:
    def __init__(self, text_processor):
        self.text_processor = text_processor
        self.severity_weights = self.load_severity_mapping()
        self.critical_conditions = self.load_critical_conditions()
        self.medication_risks = self.load_medication_risks()
        self.urgency_keywords = self.load_urgency_keywords()
        self.completeness_requirements = self.load_completeness_requirements()
    
    def load_severity_mapping(self) -> Dict[str, float]:
        return {
            'myocardial infarction': 95, 'heart attack': 95, 'cardiac arrest': 100,
            'stroke': 90, 'sepsis': 95, 'pulmonary embolism': 90, 'heart failure': 80,
            'pneumonia': 70, 'copd': 65, 'diabetes': 60, 'kidney failure': 75,
            'hypertension': 50, 'asthma': 45, 'arthritis': 30, 'depression': 40
        }
    
    def load_critical_conditions(self) -> List[str]:
        return [
            'cardiac arrest', 'myocardial infarction', 'heart attack', 'stroke',
            'sepsis', 'pulmonary embolism', 'heart failure', 'respiratory failure'
        ]
    
    def load_medication_risks(self) -> Dict[str, Dict]:
        return {
            'warfarin': {'risk_level': 80, 'monitoring': 'required'},
            'insulin': {'risk_level': 70, 'monitoring': 'required'},
            'digoxin': {'risk_level': 75, 'monitoring': 'required'},
            'metformin': {'risk_level': 40, 'monitoring': 'routine'},
            'aspirin': {'risk_level': 30, 'monitoring': 'minimal'}
        }
    
    def load_urgency_keywords(self) -> Dict[str, float]:
        return {
            'emergency': 100, 'urgent': 95, 'critical': 100, 'severe': 80,
            'acute': 75, 'stable': 20, 'improving': 15, 'routine': 10
        }
    
    def load_completeness_requirements(self) -> List[str]:
        return [
            'demographics', 'age', 'chief complaint', 'history', 'medications',
            'assessment', 'diagnosis', 'plan', 'follow-up'
        ]
    
    def calculate_comprehensive_risk_score(self, text: str, entities: Dict = None) -> Dict[str, float]:
        if entities is None:
            entities = self.text_processor.extract_medical_entities_enhanced(text)
        
        condition_risk = self.score_medical_conditions(entities, text)
        medication_risk = self.score_medications(entities, text)
        urgency_risk = self.score_urgency_indicators(text)
        completeness_score = self.score_document_completeness(text)
        interaction_risk = self.score_potential_interactions(entities)
        demographic_risk = self.score_demographic_factors(text)
        
        overall_risk = (
            condition_risk * 0.30 + medication_risk * 0.20 + urgency_risk * 0.20 +
            interaction_risk * 0.15 + demographic_risk * 0.10 + 
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
                severity = self.get_condition_severity(condition['text'].lower())
                confidence = condition.get('confidence', 0.5)
                weighted_severity = severity * confidence
                condition_score = max(condition_score, weighted_severity)
        return min(100, condition_score)
    
    def get_condition_severity(self, condition: str) -> float:
        condition = condition.lower().strip()
        if condition in self.severity_weights:
            return self.severity_weights[condition]
        for known_condition, severity in self.severity_weights.items():
            if known_condition in condition or condition in known_condition:
                return severity
        return 50.0
    
    def score_medications(self, entities: Dict, text: str) -> float:
        medication_score = 0.0
        if 'medications' in entities:
            for medication in entities['medications']:
                med_text = medication['text'].lower()
                if med_text in self.medication_risks:
                    med_risk = self.medication_risks[med_text]['risk_level']
                    confidence = medication.get('confidence', 0.5)
                    weighted_risk = med_risk * confidence
                    medication_score = max(medication_score, weighted_risk)
        return min(100, medication_score)
    
    def score_urgency_indicators(self, text: str) -> float:
        urgency_score = 0.0
        text_lower = text.lower()
        for keyword, weight in self.urgency_keywords.items():
            if keyword in text_lower:
                urgency_score = max(urgency_score, weight)
        return urgency_score
    
    def score_document_completeness(self, text: str) -> float:
        text_lower = text.lower()
        present_sections = sum(1 for req in self.completeness_requirements if req in text_lower)
        return (present_sections / len(self.completeness_requirements)) * 100
    
    def score_potential_interactions(self, entities: Dict) -> float:
        medications = []
        if 'medications' in entities:
            medications = [med['text'].lower() for med in entities['medications']]
        
        interaction_score = 0.0
        if len(medications) > 5:
            interaction_score = 40
        elif len(medications) > 10:
            interaction_score = 60
        
        return interaction_score
    
    def score_demographic_factors(self, text: str) -> float:
        demographic_score = 0.0
        age_matches = re.findall(r'age[:\s]*(\d{1,3})', text.lower())
        if age_matches:
            age = int(age_matches[0])
            if age >= 80:
                demographic_score = 60
            elif age >= 65:
                demographic_score = 40
            elif age >= 50:
                demographic_score = 20
        return demographic_score
    
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
        return avg_confidence * 100
