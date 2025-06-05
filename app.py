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
from typing import Dict, Any

# Import your pipeline components
import sys
sys.path.append('src')

try:
    from complete_pipeline import MedicalReportPipeline
except ImportError:
    st.error("Pipeline components not found. Please ensure all modules are in the 'src' directory.")
    st.stop()

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
    col1, col2 = st.columns(1)
    
    # Check if fine-tuned model exists
    fine_tuned_path = './fine_tuned_models/bart_medical/'
    fine_tuned_available = os.path.exists(fine_tuned_path)
    
    if fine_tuned_available:
        st.success("🎯 Fine-tuned BART model detected!")
        
        if st.button("🎯 Initialize Fine-tuned Pipeline", type="primary"):
            with st.spinner("Loading fine-tuned models..."):
                try:
                    st.session_state.pipeline = MedicalReportPipeline(use_fine_tuned=True)
                    st.success("✅ Fine-tuned pipeline initialized!")
                except Exception as e:
                    st.error(f"❌ Failed to initialize pipeline: {str(e)}")
        
        if st.button("📝 Initialize Standard Pipeline", type="secondary"):
            with st.spinner("Loading standard models..."):
                try:
                    st.session_state.pipeline = MedicalReportPipeline(use_fine_tuned=False)
                    st.success("✅ Standard pipeline initialized!")
                except Exception as e:
                    st.error(f"❌ Failed to initialize pipeline: {str(e)}")
    else:
        st.warning("⚠️ Fine-tuned model not found")
        st.info("💡 Run the Colab notebook to create a fine-tuned model")
        
        if st.button("📝 Initialize Standard Pipeline", type="primary"):
            with st.spinner("Loading standard models..."):
                try:
                    st.session_state.pipeline = MedicalReportPipeline(use_fine_tuned=False)
                    st.success("✅ Standard pipeline initialized!")
                except Exception as e:
                    st.error(f"❌ Failed to initialize pipeline: {str(e)}")
    
    st.markdown("---")
    
    # Model information
    if st.session_state.pipeline:
        st.subheader("📊 Model Status")
        model_info = st.session_state.pipeline.model_manager.get_model_info()
        
        # Show model type
        if model_info['summarizer'].endswith('(fine-tuned)'):
            st.success(f"🎯 {model_info['summarizer']}")
        else:
            st.info(f"📝 {model_info['summarizer']}")
            
        st.success("🏥 ClinicalBERT: Ready")
        st.success("🔤 SentenceTransformer: Ready")
        
        # Show fine-tuned model availability
        if model_info.get('fine_tuned_available', False):
            st.success("🎯 Fine-tuned model: Available")
        else:
            st.warning("⚠️ Fine-tuned model: Not found")
        
        # Model comparison button
        if model_info.get('fine_tuned_available', False) and model_info['summarizer'].endswith('(fine-tuned)'):
            if st.button("📊 Enable Model Comparison"):
                st.session_state.show_comparison = True
                st.success("✅ Comparison mode enabled!")
        
        # Processing statistics
        stats = st.session_state.pipeline.get_processing_stats()
        st.subheader("📈 Statistics")
        st.metric("Documents Processed", stats['total_documents_processed'])
        st.metric("Avg Processing Time", f"{stats['average_processing_time']:.2f}s")
        st.metric("Success Rate", f"{stats['success_rate']*100:.1f}%")
    else:
        st.warning("⚠️ Pipeline not initialized")
    
    st.markdown("---")
    
    # Settings
    st.subheader("⚙️ Settings")
    show_detailed_analysis = st.checkbox("Show detailed analysis", value=True)
    show_entity_confidence = st.checkbox("Show entity confidence", value=False)
    
    st.markdown("---")
    
    # Quick Demo Samples
    st.subheader("📋 Quick Demo Samples")
    st.info("⚠️ Copy a sample below FIRST, then initialize the AI pipeline!")
    
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
    
    with st.expander("🟡 Moderate Risk Diabetes", expanded=False):
        diabetes_sample = """Patient Demographics: Age: 45 years, Female
Chief Complaint: Routine diabetes management  
Medical History: Well-controlled diabetes mellitus type 2
Medications: Metformin 500mg twice daily
Assessment: Diabetes well-controlled
Follow-up: Primary care in 3 months"""
        
        st.code(diabetes_sample, language=None)

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
    # Document input section
    st.subheader("📝 Document Input")
    
    input_method = st.radio("Choose input method:", ["📄 Text Input", "📁 File Upload"], horizontal=True)
    
    document_text = ""
    document_id = ""
    
    if input_method == "📄 Text Input":
        col1, col2 = st.columns([3, 1])
        with col1:
            document_text = st.text_area(
                "Paste your medical document here:",
                height=200,
                placeholder="Enter medical report text..."
            )
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
    🤖 Model: {model_info['summarizer']}
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
    
    # Summaries
    st.subheader("📝 Document Summaries")
    
    summaries = result['summaries']
    
    # Show model comparison if requested
    if st.session_state.get('show_comparison', False) and st.session_state.pipeline:
        try:
            comparison = st.session_state.pipeline.model_manager.compare_models(document_text)
            
            st.markdown("### 🔍 Model Comparison")
            st.markdown('<div class="model-comparison">', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎯 Fine-tuned BART")
                st.markdown(f"""
                <div class="summary-box">
                {comparison['current_summary']}
                </div>
                """, unsafe_allow_html=True)
                st.caption(f"Length: {len(comparison['current_summary'].split())} words")
            
            with col2:
                st.markdown("#### 📝 Pre-trained BART")
                st.markdown(f"""
                <div class="summary-box">
                {comparison['pretrained_summary']}
                </div>
                """, unsafe_allow_html=True)
                st.caption(f"Length: {len(comparison['pretrained_summary'].split())} words")
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("---")
        except Exception as e:
            st.warning(f"Model comparison failed: {e}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔍 Extractive Summary")
        st.markdown(f"""
        <div class="summary-box">
        {summaries['extractive']}
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"Length: {len(summaries['extractive'].split())} words")
    
    with col2:
        st.markdown("#### 🤖 Abstractive Summary")
        st.markdown(f"""
        <div class="summary-box">
        {summaries['abstractive']}
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"Length: {len(summaries['abstractive'].split())} words")
    
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
    if st.button("📥 Download Results"):
        json_str = json.dumps(result, indent=2, default=str)
        st.download_button(
            label="📥 Download JSON Report",
            data=json_str,
            file_name=f"medical_analysis_{doc_info['document_id']}.json",
            mime="application/json"
        )

# Footer
st.markdown("---")
footer_text = "🏥 Medical Document Analyzer | Powered by BART"
if st.session_state.pipeline:
    model_info = st.session_state.pipeline.model_manager.get_model_info()
    if model_info['summarizer'].endswith('(fine-tuned)'):
        footer_text += " (Fine-tuned)"
    else:
        footer_text += " (Pre-trained)"

footer_text += ", ClinicalBERT & SentenceTransformers"

st.markdown(f"""
<div style="text-align: center; color: #666; padding: 1rem;">
    {footer_text}<br>
    Built with Streamlit | AI-Powered Medical Text Analysis
</div>
""", unsafe_allow_html=True)
