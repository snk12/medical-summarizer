import torch
from transformers import pipeline, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

class ModelManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        self.load_all_models()
    
    def load_all_models(self):
        """Load all required models"""
        print("📥 Loading AI models...")
        
        self.models['sentence_transformer'] = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.models['bart_summarizer'] = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=0 if self.device == "cuda" else -1,
            max_length=150,
            min_length=50,
            do_sample=False
        )
        
        self.models['medical_ner'] = pipeline(
            "ner",
            model="emilyalsentzer/Bio_ClinicalBERT",
            device=0 if self.device == "cuda" else -1,
            aggregation_strategy="simple"
        )
        
        self.models['general_ner'] = pipeline(
            "ner",
            model="dbmdz/bert-large-cased-finetuned-conll03-english",
            device=0 if self.device == "cuda" else -1,
            aggregation_strategy="simple"
        )
        
        print("✅ All models loaded successfully!")
    
    def get_sentence_embeddings(self, sentences):
        return self.models['sentence_transformer'].encode(sentences)
    
    def summarize_text(self, text, max_length=150, min_length=50):
        try:
            max_input_length = 1024
            if len(text.split()) > max_input_length:
                text = ' '.join(text.split()[:max_input_length])
            
            result = self.models['bart_summarizer'](
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            return result[0]['summary_text']
        except Exception as e:
            return f"Error generating summary: {e}"
    
    def extract_medical_entities(self, text):
        try:
            entities = self.models['medical_ner'](text)
            processed_entities = []
            for entity in entities:
                processed_entities.append({
                    'text': entity['word'],
                    'label': entity['entity_group'],
                    'confidence': entity['score'],
                    'start': entity.get('start', 0),
                    'end': entity.get('end', 0)
                })
            return processed_entities
        except Exception as e:
            return []
    
    def extract_general_entities(self, text):
        try:
            entities = self.models['general_ner'](text)
            processed_entities = []
            for entity in entities:
                processed_entities.append({
                    'text': entity['word'],
                    'label': entity['entity_group'],
                    'confidence': entity['score']
                })
            return processed_entities
        except Exception as e:
            return []
    
    def get_model_info(self):
        return {
            'device': self.device,
            'models_loaded': list(self.models.keys()),
            'sentence_transformer': 'all-MiniLM-L6-v2',
            'summarizer': 'facebook/bart-large-cnn',
            'medical_ner': 'emilyalsentzer/Bio_ClinicalBERT',
            'general_ner': 'dbmdz/bert-large-cased-finetuned-conll03-english'
        }
