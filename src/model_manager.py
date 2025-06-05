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
        
        # Your Hugging Face model ID
        self.hf_model_id = "swapnilkharade/bart-medical-summarizer"
        
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
    
    def download_model_from_hf(self):
        """Download model from Hugging Face Hub"""
        try:
            # Try to import streamlit for UI feedback
            try:
                import streamlit as st
                st.info("🔄 Downloading fine-tuned model from Hugging Face...")
                spinner_context = st.spinner("Downloading model files...")
            except ImportError:
                # If not in Streamlit environment, just print
                print("🔄 Downloading fine-tuned model from Hugging Face...")
                spinner_context = None
            
            # Create directory
            os.makedirs(self.fine_tuned_path, exist_ok=True)
            
            # Download using transformers
            from transformers import BartForConditionalGeneration, BartTokenizer
            
            if spinner_context:
                with spinner_context:
                    model = BartForConditionalGeneration.from_pretrained(self.hf_model_id)
                    tokenizer = BartTokenizer.from_pretrained(self.hf_model_id)
                    
                    # Save locally
                    model.save_pretrained(self.fine_tuned_path)
                    tokenizer.save_pretrained(self.fine_tuned_path)
            else:
                model = BartForConditionalGeneration.from_pretrained(self.hf_model_id)
                tokenizer = BartTokenizer.from_pretrained(self.hf_model_id)
                
                # Save locally
                model.save_pretrained(self.fine_tuned_path)
                tokenizer.save_pretrained(self.fine_tuned_path)
            
            try:
                import streamlit as st
                st.success("✅ Model downloaded successfully!")
            except ImportError:
                print("✅ Model downloaded successfully!")
            
            return True
            
        except Exception as e:
            error_msg = f"❌ Failed to download model: {e}"
            try:
                import streamlit as st
                st.error(error_msg)
            except ImportError:
                print(error_msg)
            return False
    
    def load_bart_model(self):
        """Load BART model (fine-tuned if available, otherwise pre-trained)"""
        
        if self.use_fine_tuned:
            # Check if model exists locally
            if os.path.exists(self.fine_tuned_path) and os.listdir(self.fine_tuned_path):
                try:
                    try:
                        import streamlit as st
                        with st.spinner(f"Loading fine-tuned BART from {self.fine_tuned_path}..."):
                            pass
                    except ImportError:
                        pass
                    
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
                    
                    try:
                        import streamlit as st
                        st.success("✅ Fine-tuned BART loaded from local files!")
                    except ImportError:
                        pass
                    
                    return
                except Exception as e:
                    print(f"⚠️ Failed to load local model: {e}")
            
            # Try to download from Hugging Face
            print("📥 Local model not found, trying to download from Hugging Face...")
            if self.download_model_from_hf():
                try:
                    try:
                        import streamlit as st
                        with st.spinner("Loading downloaded model..."):
                            pass
                    except ImportError:
                        pass
                    
                    self.models['bart_summarizer'] = pipeline(
                        "summarization",
                        model=self.fine_tuned_path,
                        device=0 if self.device == "cuda" else -1,
                        max_length=150,
                        min_length=50,
                        do_sample=False
                    )
                    self.model_type = "fine-tuned"
                    print("✅ Downloaded fine-tuned BART loaded!")
                    
                    try:
                        import streamlit as st
                        st.success("✅ Fine-tuned BART loaded from downloaded files!")
                    except ImportError:
                        pass
                    
                    return
                except Exception as e:
                    print(f"❌ Failed to load downloaded model: {e}")
            
            # Try to load directly from Hugging Face (without local download)
            try:
                try:
                    import streamlit as st
                    with st.spinner("Loading fine-tuned model directly from Hugging Face..."):
                        pass
                    st.info("📥 Loading your fine-tuned model from Hugging Face...")
                except ImportError:
                    pass
                
                print("📥 Loading fine-tuned model directly from Hugging Face...")
                self.models['bart_summarizer'] = pipeline(
                    "summarization",
                    model=self.hf_model_id,
                    device=0 if self.device == "cuda" else -1,
                    max_length=150,
                    min_length=50,
                    do_sample=False
                )
                self.model_type = "fine-tuned (HF)"
                print("✅ Fine-tuned BART loaded from Hugging Face!")
                
                try:
                    import streamlit as st
                    st.success("✅ Fine-tuned BART loaded from Hugging Face!")
                except ImportError:
                    pass
                
                return
            except Exception as e:
                print(f"⚠️ Failed to load from Hugging Face: {e}")
                try:
                    import streamlit as st
                    st.warning(f"⚠️ Failed to load from Hugging Face: {e}")
                    st.info("🔄 Falling back to pre-trained model...")
                except ImportError:
                    pass
        
        # Load pre-trained model as fallback
        try:
            try:
                import streamlit as st
                with st.spinner("Loading pre-trained BART model..."):
                    pass
            except ImportError:
                pass
            
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
            
            try:
                import streamlit as st
                st.success("✅ Pre-trained BART loaded successfully!")
            except ImportError:
                pass
                
        except Exception as e:
            error_msg = f"❌ BART model failed to load: {e}"
            print(error_msg)
            try:
                import streamlit as st
                st.error(error_msg)
            except ImportError:
                pass
            self.models['bart_summarizer'] = None
    
    def switch_to_fine_tuned(self, model_path=None):
        """Switch to fine-tuned model during runtime"""
        if model_path is None:
            model_path = self.fine_tuned_path
            
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
            if 'bart_summarizer' not in self.models or self.models['bart_summarizer'] is None:
                raise Exception("BART model not available")
            
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
    
    def extract_general_entities(self, text):
        try:
            if 'general_ner' not in self.models:
                return []
            
            entities = self.models['general_ner'](text)
            processed_entities = []
            for entity in entities:
                processed_entities.append({
                    'text': entity.get('word', '').replace('##', ''),
                    'label': entity.get('entity_group', 'UNKNOWN'),
                    'confidence': entity.get('score', 0.5)
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
            'summarizer_path': self.hf_model_id if self.model_type.startswith("fine-tuned") else "facebook/bart-large-cnn",
            'medical_ner': 'emilyalsentzer/Bio_ClinicalBERT',
            'general_ner': 'dbmdz/bert-large-cased-finetuned-conll03-english',
            'fine_tuned_available': os.path.exists(self.fine_tuned_path) and bool(os.listdir(self.fine_tuned_path)) if os.path.exists(self.fine_tuned_path) else False,
            'hf_model_id': self.hf_model_id
        }
    
    def compare_models(self, text, max_length=150):
        """Compare pre-trained vs fine-tuned model outputs"""
        if not hasattr(self, '_pretrained_pipeline'):
            # Load pre-trained for comparison
            try:
                self._pretrained_pipeline = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    device=0 if self.device == "cuda" else -1,
                    max_length=max_length,
                    min_length=50
                )
            except Exception as e:
                print(f"Failed to load comparison model: {e}")
                return None
        
        # Get summary from current model
        current_summary = self.summarize_text(text, max_length)
        
        # Get summary from pre-trained model
        try:
            pretrained_result = self._pretrained_pipeline(text, max_length=max_length, min_length=50)
            pretrained_summary = pretrained_result[0]['summary_text']
        except Exception as e:
            pretrained_summary = f"Error generating pre-trained summary: {e}"
        
        return {
            'current_model': self.model_type,
            'current_summary': current_summary,
            'pretrained_summary': pretrained_summary,
            'model_path': self.hf_model_id if self.model_type.startswith("fine-tuned") else "facebook/bart-large-cnn"
        }
    
    def test_fine_tuned_model(self):
        """Test if the fine-tuned model is working"""
        test_text = """Patient Demographics: 65-year-old male
Chief Complaint: Chest pain for 3 hours
Assessment: Acute myocardial infarction
Plan: Emergency catheterization"""
        
        try:
            if self.model_type.startswith("fine-tuned"):
                summary = self.summarize_text(test_text)
                print(f"✅ Fine-tuned model test successful!")
                print(f"Test summary: {summary}")
                return True
            else:
                print("⚠️ Fine-tuned model not loaded")
                return False
        except Exception as e:
            print(f"❌ Fine-tuned model test failed: {e}")
            return False
