import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import time
import io
import base64
import os
import re
from typing import Dict, Any

# Import your pipeline components
import sys
sys.path.append('src')

try:
    from complete_pipeline import MedicalReportPipeline
except ImportError:
    st.error("Pipeline components not found. Please ensure all modules are in the 'src' directory.")
    st.stop()

# Helper functions
def display_model_badge(model_type):
    """Display a nice badge for the current model"""
    if "fine-tuned" in model_type.lower():
        return "🎯 Fine-tuned"
    else:
        return "📝 Standard"

def get_model_color(model_type):
    """Get color scheme for model type"""
    if "fine-tuned" in model_type.lower():
        return "#28a745"  # Green
    else:
        return "#6c757d"  # Gray

def clean_summary_text(text):
    """Clean up summary text formatting"""
    if not text:
        return text
    
    # Fix spacing issues from your model
    text = re.sub(r'(\w+)([A-Z])', r'\1. \2', text)
    text = re.sub(r'(male|female)(Chief)', r'\1. Chief', text)
    text = re.sub(r'(Assessment|Plan|History):', r'. \1:', text)
    
    # Clean up extra spaces and periods
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\.+', '.', text)
    
    # Ensure proper ending
    if not text.endswith('.'):
        text += '.'
    
    return text

# Page configuration
st.set_page_config(
    page_title="Medical Document Analyzer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .risk-critical { background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%); }
    .risk-high { background: linear-gradient(135deg, #ffa726 0%, #ff9800 100%); }
    .risk-moderate { background: linear-gradient(135deg, #ffeb3b 0%, #fbc02d 100%); color: #333; }
    .risk-low { background: linear-gradient(135deg, #66bb6a 0%, #4caf50 100%); }
    .risk-minimal { background: linear-gradient(135deg, #81c784 0%, #66bb6a 100%); }
    
    .entity-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        margin: 0.25rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    .entity-conditions { background: #ffebee; color: #c62828; }
    .entity-medications { background: #e8f5e8; color: #2e7d32; }
    .entity-procedures { background: #e3f2fd; color: #1565c0; }
    .entity-general { background: #f3e5f5; color: #6a1b9a; }
    
    .summary-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    
    .processing-time {
        background: #e8f5e8;
        padding: 0.5rem;
        border-radius: 5px;
        color: #2e7d32;
        font-weight: bold;
        text-align: center;
    }
    
    .model-comparison {
        background: #f0f8ff;
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
    }
    
    .model-badge-finetuned {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.25rem;
    }
    
    .model-badge-standard {
        background: linear-gradient(135deg, #6c757d 0%, #adb5bd 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
    st.session_state.processing_history = []
    st.session_state.current_result = None
    st.session_state.show_comparison = False

# Header
st.markdown('<h1 class="main-header">🏥 Medical Document Analyzer</h1>', unsafe_allow_html=True)
st.markdown("### AI-Powered Medical Report Summarization with Risk Assessment")

# Sidebar
with st.sidebar:
    st.header("🛠️ Configuration")
    
    # Pipeline initialization
    st.subheader("🚀 Initialize Pipeline")
    
    # Check if fine-tuned model exists locally
    fine_tuned_path = './fine_tuned_models/bart_medical/'
    fine_tuned_available = os.path.exists(fine_tuned_path) and bool(os.listdir(fine_tuned_path)) if os.path.exists(fine_tuned_path) else False
    
    # Show model availability status
    if fine_tuned_available:
        st.success("🎯 Fine-tuned model detected locally!")
    else:
        st.info("🌐 Fine-tuned model will be downloaded from Hugging Face")
    
    # Model selection buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎯 Fine-tuned", type="primary", help="Uses your custom medical model", use_container_width=True):
            with st.spinner("Loading fine-tuned models..."):
                try:
                    st.session_state.pipeline = MedicalReportPipeline(use_fine_tuned=True)
                    st.success("✅ Fine-tuned pipeline ready!")
                    st.balloons()
                    time.sleep(1)  # Brief pause to show success
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed: {str(e)}")
    
    with col2:
        if st.button("📝 Standard", type="secondary", help="Uses pre-trained BART model", use_container_width=True):
            with st.spinner("Loading standard models..."):
                try:
                    st.session_state.pipeline = MedicalReportPipeline(use_fine_tuned=False)
                    st.success("✅ Standard pipeline ready!")
                    time.sleep(1)  # Brief pause to show success
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed: {str(e)}")
    
    st.markdown("---")
    
    # Model information
    if st.session_state.pipeline:
        st.subheader("📊 Model Status")
        model_info = st.session_state.pipeline.model_manager.get_model_info()
        
        # Show current model with styling
        model_type = model_info['summarizer']
        if "fine-tuned" in model_type.lower():
            st.markdown('<div class="model-badge-finetuned">🎯 Fine-tuned BART Active</div>', unsafe_allow_html=True)
            if 'hf_model_id' in model_info:
                st.caption(f"📍 {model_info['hf_model_id']}")
        else:
            st.markdown('<div class="model-badge-standard">📝 Standard BART Active</div>', unsafe_allow_html=True)
            st.caption("📍 facebook/bart-large-cnn")
        
        # Other models status
        st.success("🏥 ClinicalBERT: Ready")
        st.success("🔤 SentenceTransformer: Ready")
        
        # Model comparison option
        if "fine-tuned" in model_type.lower():
            if st.button("📊 Enable Comparison", help="Compare fine-tuned vs pre-trained"):
                st.session_state.show_comparison = True
                st.success("✅ Comparison mode enabled!")
        
        # Processing statistics
        stats = st.session_state.pipeline.get_processing_stats()
        st.subheader("📈 Statistics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", stats['total_documents_processed'])
            st.metric("Success Rate", f"{stats['success_rate']*100:.1f}%")
        with col2:
            st.metric("Avg Time", f"{stats['average_processing_time']:.2f}s")
            model_badge = display_model_badge(model_type)
            st.metric("Model", model_badge)
        
    else:
        st.warning("⚠️ Pipeline not initialized")
        st.info("👆 Click a button above to start")
    
    st.markdown("---")
    
    # Settings
    st.subheader("⚙️ Settings")
    show_detailed_analysis = st.checkbox("Show detailed analysis", value=True)
    show_entity_confidence = st.checkbox("Show entity confidence", value=False)
    
    st.markdown("---")
    
    # Quick Demo Samples
    st.subheader("📋 Quick Demo Samples")
    st.info("💡 Copy a sample below, then initialize AI pipeline!")
    
    with st.expander("🔴 High Risk Stroke Patient", expanded=False):
        stroke_sample = """PATIENT DISCHARGE SUMMARY

Patient Demographics:
Age: 60 years
Gender: Male
Date of Admission: 2025-5-29
Date of Discharge: 2025-5-30

Chief Complaint:
Patient presented with symptoms related to stroke.

Medical History:
Patient has a known history of stroke. Patient reports compliance with prescribed medications.

Medications:
- Metformin 25mg twice daily

Procedures Performed:
- Ekg: shows improvement
- Colonoscopy: normal

Assessment and Plan:
Patient's stroke is improving. Continue current treatment plan with modifications as noted.

Discharge Instructions:
Continue medications as prescribed. Monitor symptoms and return if worsening. Follow up with primary care physician.

Follow-up:
Follow up with primary care in 3 months."""
        
        st.code(stroke_sample, language=None)
        if st.button("📋 Copy Stroke Sample", key="copy_stroke"):
            st.session_state.demo_text = stroke_sample
    
    with st.expander("🟡 Moderate Risk Diabetes", expanded=False):
        diabetes_sample = """Patient Demographics: Age: 45 years, Female
Chief Complaint: Routine diabetes management  
Medical History: Well-controlled diabetes mellitus type 2
Medications: Metformin 500mg twice daily
Assessment: Diabetes well-controlled
Follow-up: Primary care in 3 months"""
        
        st.code(diabetes_sample, language=None)
        if st.button("📋 Copy Diabetes Sample", key="copy_diabetes"):
            st.session_state.demo_text = diabetes_sample

# Main content area
if not st.session_state.pipeline:
    st.info("👈 Please initialize the AI pipeline in the sidebar to begin.")
    
    # Show sample document preview
    st.subheader("📄 Sample Medical Document")
    sample_text = """PATIENT DISCHARGE SUMMARY

Patient Demographics:
Age: 72 years
Gender: Male
Date of Admission: 2024-03-15
Date of Discharge: 2024-03-18

Chief Complaint:
Patient presented with chest pain and shortness of breath.

Medical History:
Patient has a known history of diabetes mellitus and hypertension. Patient reports compliance with prescribed medications.

Medications:
- Metformin 500mg twice daily
- Lisinopril 10mg daily
- Aspirin 81mg daily

Procedures Performed:
- Electrocardiogram: shows improvement
- Chest X-ray: within normal limits

Assessment and Plan:
Patient's diabetes mellitus is well-controlled. Continue current treatment plan with modifications as noted.

Discharge Instructions:
Continue medications as prescribed. Monitor symptoms and return if worsening. Maintain healthy diet and exercise as tolerated.

Follow-up:
Follow up with primary care in 2 weeks."""
    
    st.text_area("Sample document:", value=sample_text, height=300, disabled=True)
    st.info("This is what a typical medical document looks like. The AI will extract entities, generate summaries, and assess risk levels.")

else:
    # Show current model status
    model_info = st.session_state.pipeline.model_manager.get_model_info()
    if "fine-tuned" in model_info['summarizer'].lower():
        st.success("🎯 **Fine-tuned Model Active** - Optimized for medical text analysis")
    else:
        st.info("📝 **Standard Model Active** - General purpose summarization")
    
    # Document input section
    st.subheader("📝 Document Input")
    
    input_method = st.radio("Choose input method:", ["📄 Text Input", "📁 File Upload"], horizontal=True)
    
    document_text = ""
    document_id = ""
    
    if input_method == "📄 Text Input":
        col1, col2 = st.columns([3, 1])
        with col1:
            # Use demo text if available
            default_text = st.session_state.get('demo_text', '')
            document_text = st.text_area(
                "Paste your medical document here:",
                value=default_text,
                height=200,
                placeholder="Enter medical report text..."
            )
            # Clear demo text after use
            if 'demo_text' in st.session_state:
                del st.session_state.demo_text
                
        with col2:
            document_id = st.text_input("Document ID (optional):", placeholder="e.g., PAT001")
            
    else:  # File Upload
        uploaded_file = st.file_uploader(
            "Upload medical document",
            type=['txt'],
            help="Supported formats: TXT"
        )
        
        if uploaded_file:
            document_id = uploaded_file.name.split('.')[0]
            document_text = str(uploaded_file.read(), "utf-8")
    
    # Processing button
    if st.button("🔍 Analyze Document", type="primary", disabled=not document_text.strip()):
        if document_text.strip():
            with st.spinner("🤖 Processing document... This may take a few seconds."):
                start_time = time.time()
                
                # Process the document
                result = st.session_state.pipeline.process_document(
                    document_text, 
                    document_id or f"doc_{int(start_time)}"
                )
                
                processing_time = time.time() - start_time
                
                # Store result
                st.session_state.current_result = result
                st.session_state.processing_history.append({
                    'timestamp': datetime.now(),
                    'document_id': result['document_info']['document_id'],
                    'success': result['processing_metadata']['success'],
                    'processing_time': processing_time
                })
                
                if result['processing_metadata']['success']:
                    st.success(f"✅ Document processed successfully in {processing_time:.2f} seconds!")
                else:
                    st.error(f"❌ Processing failed: {result['processing_metadata']['error_message']}")

# Results display
if st.session_state.current_result and st.session_state.current_result['processing_metadata']['success']:
    result = st.session_state.current_result
    
    st.markdown("---")
    st.subheader("📊 Analysis Results")
    
    # Processing info
    doc_info = result['document_info']
    model_info = st.session_state.pipeline.model_manager.get_model_info()
    
    st.markdown(f"""
    <div class="processing-time">
    📄 Document: {doc_info['document_id']} | 
    ⏱️ Processed in {doc_info['processing_time_seconds']:.2f}s | 
    📏 {doc_info['original_length']} characters |
    🤖 Model: {display_model_badge(model_info['summarizer'])}
    </div>
    """, unsafe_allow_html=True)
    
    # Risk Assessment
    risk_scores = result['risk_assessment']['scores']
    
    st.subheader("🎯 Risk Assessment")
    
    # Risk level indicator
    risk_category = risk_scores['risk_category'].lower()
    risk_score = risk_scores['overall_risk']
    
    st.markdown(f"""
    <div class="metric-card risk-{risk_category}">
        <h3>Overall Risk Level</h3>
        <h1>{risk_scores['risk_category']}</h1>
        <h2>{risk_score:.1f}/100</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Risk breakdown
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🏥 Condition Severity", f"{risk_scores['condition_severity']:.1f}/100")
        st.metric("💊 Medication Risk", f"{risk_scores['medication_risk']:.1f}/100")
    
    with col2:
        st.metric("🚨 Urgency Level", f"{risk_scores['urgency_level']:.1f}/100")
        st.metric("⚡ Interaction Risk", f"{risk_scores['interaction_risk']:.1f}/100")
    
    with col3:
        st.metric("👤 Demographic Risk", f"{risk_scores['demographic_risk']:.1f}/100")
        st.metric("🎯 Confidence Score", f"{risk_scores['confidence_score']:.1f}/100")
    
    # Risk gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Risk Score"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 20], 'color': "lightgray"},
                {'range': [20, 40], 'color': "gray"},
                {'range': [40, 60], 'color': "yellow"},
                {'range': [60, 80], 'color': "orange"},
                {'range': [80, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig_gauge.update_layout(height=300)
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Summaries section
    st.subheader("📝 Document Summaries")
    summaries = result['summaries']
    
    # Model comparison if enabled
    if st.session_state.get('show_comparison', False) and st.session_state.pipeline:
        try:
            comparison = st.session_state.pipeline.model_manager.compare_models(document_text)
            
            if comparison:
                st.subheader("🔬 Model Comparison Analysis")
                
                # Create tabs for better organization
                tab1, tab2, tab3 = st.tabs(["📊 Side-by-Side", "📈 Analysis", "🎯 Recommendations"])
                
                with tab1:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 🎯 Fine-tuned BART")
                        cleaned_current = clean_summary_text(comparison['current_summary'])
                        st.markdown(f"""
                        <div class="summary-box" style="border-left: 4px solid #28a745;">
                        {cleaned_current}
                        </div>
                        """, unsafe_allow_html=True)
                        st.caption(f"📏 Length: {len(cleaned_current.split())} words")
                    
                    with col2:
                        st.markdown("#### 📝 Pre-trained BART")
                        st.markdown(f"""
                        <div class="summary-box" style="border-left: 4px solid #6c757d;">
                        {comparison['pretrained_summary']}
                        </div>
                        """, unsafe_allow_html=True)
                        st.caption(f"📏 Length: {len(comparison['pretrained_summary'].split())} words")
                
                with tab2:
                    # Analysis metrics
                    fine_words = comparison['current_summary'].lower().split()
                    pre_words = comparison['pretrained_summary'].lower().split()
                    
                    # Count medical terms
                    medical_terms = ['patient', 'diagnosis', 'treatment', 'medication', 'procedure', 'symptoms', 'condition', 'diabetes', 'hypertension', 'chest', 'pain']
                    fine_medical = sum(1 for term in medical_terms if term in fine_words)
                    pre_medical = sum(1 for term in medical_terms if term in pre_words)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Fine-tuned Medical Terms", fine_medical, delta=fine_medical-pre_medical)
                    with col2:
                        st.metric("Pre-trained Medical Terms", pre_medical)
                    with col3:
                        improvement = "Better" if fine_medical > pre_medical else "Similar" if fine_medical == pre_medical else "Needs Work"
                        st.metric("Assessment", improvement)
                
                with tab3:
                    if fine_medical > pre_medical:
                        st.success("🎯 **Fine-tuned model is performing better!**")
                        st.write("✅ Uses more medical terminology")
                        st.write("✅ Better structured for medical context")
                        st.write("✅ Recommended for medical document analysis")
                    elif fine_medical == pre_medical:
                        st.info("🤔 **Models perform similarly**")
                        st.write("• Medical term usage is comparable")
                        st.write("• Fine-tuned model may show benefits in other aspects")
                        st.write("• Consider testing with more complex medical documents")
                    else:
                        st.warning("⚠️ **Fine-tuned model may need more training**")
                        st.write("• Consider more training epochs")
                        st.write("• Add more diverse medical training data")
                        st.write("• Fine-tune hyperparameters")
                
                st.markdown("---")
        except Exception as e:
            st.warning(f"Model comparison unavailable: {e}")
    
    # Regular summaries
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔍 Extractive Summary")
        st.markdown(f"""
        <div class="summary-box">
        {summaries['extractive']}
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"📏 Length: {len(summaries['extractive'].split())} words")
    
    with col2:
        st.markdown("#### 🤖 Abstractive Summary")
        cleaned_abstractive = clean_summary_text(summaries['abstractive'])
        st.markdown(f"""
        <div class="summary-box">
        {cleaned_abstractive}
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"📏 Length: {len(cleaned_abstractive.split())} words")
    
    # Entities
    st.subheader("🔍 Extracted Medical Entities")
    
    entities = result['entities']
    
    # Entity summary
    entity_counts = {category: len(entity_list) for category, entity_list in entities.items() if entity_list}
    
    if entity_counts:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🏥 Conditions", entity_counts.get('conditions', 0))
        with col2:
            st.metric("💊 Medications", entity_counts.get('medications', 0))
        with col3:
            st.metric("🔬 Procedures", entity_counts.get('procedures', 0))
        with col4:
            st.metric("📋 General", entity_counts.get('general', 0))
        
        # Entity details
        for category, entity_list in entities.items():
            if entity_list:
                st.markdown(f"#### {category.title()}")
                
                entity_html = ""
                for entity in entity_list:
                    confidence_text = f" ({entity['confidence']:.2f})" if show_entity_confidence else ""
                    entity_html += f'<span class="entity-badge entity-{category}">{entity["text"]}{confidence_text}</span>'
                
                st.markdown(entity_html, unsafe_allow_html=True)
    else:
        st.info("No entities extracted from this document.")
    
    # Download results
    st.subheader("📥 Export Results")
    col1, col2 = st.columns(2)
    
    with col1:
        json_str = json.dumps(result, indent=2, default=str)
        st.download_button(
            label="📥 Download JSON Report",
            data=json_str,
            file_name=f"medical_analysis_{doc_info['document_id']}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        # Create a simple text report
        text_report = f"""MEDICAL DOCUMENT ANALYSIS REPORT
        
Document ID: {doc_info['document_id']}
Processed: {doc_info['processed_at']}
Model: {display_model_badge(model_info['summarizer'])}

RISK ASSESSMENT:
Overall Risk: {risk_scores['risk_category']} ({risk_score:.1f}/100)
- Condition Severity: {risk_scores['condition_severity']:.1f}/100
- Medication Risk: {risk_scores['medication_risk']:.1f}/100
- Urgency Level: {risk_scores['urgency_level']:.1f}/100

SUMMARIES:
Extractive: {summaries['extractive']}

Abstractive: {cleaned_abstractive}

ENTITIES FOUND:
{chr(10).join([f"- {category.title()}: {len(entity_list)} items" for category, entity_list in entities.items() if entity_list])}
"""
        
        st.download_button(
            label="📄 Download Text Report",
            data=text_report,
            file_name=f"medical_report_{doc_info['document_id']}.txt",
            mime="text/plain",
            use_container_width=True
        )

# Footer
st.markdown("---")
footer_text = "🏥 Medical Document Analyzer | Powered by "
if st.session_state.pipeline:
    model_info = st.session_state.pipeline.model_manager.get_model_info()
    if "fine-tuned" in model_info['summarizer'].lower():
        footer_text += "Fine-tuned BART"
    else:
        footer_text += "BART (Pre-trained)"
else:
    footer_text += "BART"

footer_text += ", ClinicalBERT & SentenceTransformers"

st.markdown(f"""
<div style="text-align: center; color: #666; padding: 1rem;">
    {footer_text}<br>
    Built with Streamlit | AI-Powered Medical Text Analysis<br>
    🌟 <a href="https://huggingface.co/swapnilkharade/bart-medical-summarizer" target="_blank">View Fine-tuned Model on Hugging Face</a>
</div>
""", unsafe_allow_html=True)
