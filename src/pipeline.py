import time
import json
from datetime import datetime
from typing import Dict, Any, Optional
import traceback

from src.model_manager import ModelManager
from src.text_processor import MedicalTextProcessor  
from src.summarizer import MedicalSummarizer
from src.risk_scorer import MedicalRiskScorer

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
            risk_explanation = self.risk_scorer.generate_risk_explanation(risk_scores, entities, processed_text)
            
            processing_time = time.time() - start_time
            
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
                'summaries': {
                    'extractive': summaries['extractive'],
                    'abstractive': summaries['abstractive'],
                    'comparison': summaries['method_comparison']
                },
                'risk_assessment': {
                    'scores': risk_scores,
                    'explanation': risk_explanation
                },
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
                    'success': False,
                    'error_message': str(e)
                }
            }
    
    def _update_processing_stats(self, processing_time: float, success: bool):
        self.processing_stats['total_documents_processed'] += 1
        
        prev_avg = self.processing_stats['average_processing_time']
        n = self.processing_stats['total_documents_processed']
        self.processing_stats['average_processing_time'] = (prev_avg * (n-1) + processing_time) / n
        
        if success:
            prev_success_count = self.processing_stats['success_rate'] * (n-1)
            self.processing_stats['success_rate'] = (prev_success_count + 1) / n
        else:
            prev_success_count = self.processing_stats['success_rate'] * (n-1)
            self.processing_stats['success_rate'] = prev_success_count / n
        
        self.processing_stats['last_updated'] = datetime.now().isoformat()
    
    def get_processing_stats(self) -> Dict:
        return self.processing_stats.copy()
