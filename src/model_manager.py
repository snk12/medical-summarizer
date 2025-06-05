import torch
import os
from transformers import pipeline, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

class ModelManager:
    def __init__(self, use_fine_tuned=False, fine_tuned_path='./fine_tuned_models/bart_medical/'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        self.use_fine_tuned = use_fine_tuned
        self.fine_tuned_path = fine_tuned_path
        self.model_type = "pre-trained"  # Default
        self.load_all_models()
    
    def load_all_models(self):
        """Load all required models"""
        print("📥 Loading AI models...")
        
        # Load sentence transformer
        self.models['sentence_transformer'] = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load BART summarizer (fine-tuned or pre-trained)
        self.load_bart_model()
        
        # Load medical NER
        self.models['medical_ner'] = pipeline(
            "ner",
            model="emilyalsentzer/Bio_ClinicalBERT",
            device=0 if self.device == "cuda" else -1,
            aggregation_strategy="simple"
        )
        
        # Load general NER
        self.models['general_ner'] = pipeline(
            "ner",
            model="dbmdz/bert-large-cased-finetuned-conll03-english",
            device=0 if self.device == "cuda" else -1,
            aggregation_strategy="simple"
        )
        
        print("✅ All models loaded successfully!")
    
    def load_bart_model(self):
        """Load BART model (fine-tuned if available, otherwise pre-trained)"""
        if self.use_fine_tuned and os.path.exists(self.fine_tuned_path):
            try:
                print(f"🎯 Loading fine-tuned BART from {self.fine_tuned_path}")
                self.models['bart_summarizer'] = pipeline(
                    "summarization",
                    model=self.fine_tuned_path,
                    device=0 if self.device == "cuda" else -1,
                    max_length=150,
                    min_length=50,
                    do_sample=False
                )
                self.model_type = "fine-tuned"
                print("✅ Fine-tuned BART loaded successfully!")
                return
            except Exception as e:
                print(f"⚠️ Failed to load fine-tuned model: {e}")
                print("🔄 Falling back to pre-trained model...")
        
        # Load pre-trained model
        print("📥 Loading pre-trained BART model...")
        self.models['bart_summarizer'] = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=0 if self.device == "cuda" else -1,
            max_length=150,
            min_length=50,
            do_sample=False
        )
        self.model_type = "pre-trained"
        print("✅ Pre-trained BART loaded successfully!")
    
    def switch_to_fine_tuned(self, model_path):
        """Switch to fine-tuned model during runtime"""
        if os.path.exists(model_path):
            try:
                print(f"🔄 Switching to fine-tuned model: {model_path}")
                self.models['bart_summarizer'] = pipeline(
                    "summarization",
                    model=model_path,
                    device=0 if self.device == "cuda" else -1,
                    max_length=150,
                    min_length=50,
                    do_sample=False
                )
                self.model_type = "fine-tuned"
                self.fine_tuned_path = model_path
                print("✅ Successfully switched to fine-tuned model!")
                return True
            except Exception as e:
                print(f"❌ Failed to switch model: {e}")
                return False
        else:
            print(f"❌ Model path not found: {model_path}")
            return False
    
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
            'summarizer': f'BART ({self.model_type})',
            'summarizer_path': self.fine_tuned_path if self.model_type == "fine-tuned" else "facebook/bart-large-cnn",
            'medical_ner': 'emilyalsentzer/Bio_ClinicalBERT',
            'general_ner': 'dbmdz/bert-large-cased-finetuned-conll03-english',
            'fine_tuned_available': os.path.exists(self.fine_tuned_path) if hasattr(self, 'fine_tuned_path') else False
        }
    
    def compare_models(self, text, max_length=150):
        """Compare pre-trained vs fine-tuned model outputs"""
        if not hasattr(self, '_pretrained_pipeline'):
            # Load pre-trained for comparison
            self._pretrained_pipeline = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=0 if self.device == "cuda" else -1,
                max_length=max_length,
                min_length=50
            )
        
        # Get summary from current model
        current_summary = self.summarize_text(text, max_length)
        
        # Get summary from pre-trained model
        try:
            pretrained_result = self._pretrained_pipeline(text, max_length=max_length, min_length=50)
            pretrained_summary = pretrained_result[0]['summary_text']
        except:
            pretrained_summary = "Error generating pre-trained summary"
        
        return {
            'current_model': self.model_type,
            'current_summary': current_summary,
            'pretrained_summary': pretrained_summary,
            'model_path': self.fine_tuned_path if self.model_type == "fine-tuned" else "facebook/bart-large-cnn"
        }
