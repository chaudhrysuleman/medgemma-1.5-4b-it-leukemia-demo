"""
🩸 LeukemiaScope Agentic - Multi-Agent Blood Cell Analysis
HuggingFace Spaces deployment with multi-step patient flow and PDF export
"""

import os
import sys
import gradio as gr
from PIL import Image
from datetime import datetime

# Add current dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph.workflow import run_analysis
from tools.medgemma_predictor import get_predictor
from tools.pdf_generator import generate_pdf_report

# Global state
current_patient = {}


def validate_patient_info(name, dob, gender):
    """Validate patient information before proceeding"""
    if not name or len(name.strip()) < 2:
        return False, "Please enter patient name (at least 2 characters)"
    return True, ""


def save_patient_info(name, dob, gender):
    """Save patient info and move to next step"""
    global current_patient
    
    is_valid, error_msg = validate_patient_info(name, dob, gender)
    if not is_valid:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            f"❌ {error_msg}"
        )
    
    current_patient = {
        "name": name.strip(),
        "dob": dob,
        "gender": gender,
        "id": f"LS-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    }
    
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
        f"✅ Patient registered: {name}"
    )


def go_back_to_step1():
    """Go back to patient info step"""
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
    )


def analyze_image_workflow(image):
    """Run the agentic workflow on uploaded image"""
    global current_patient
    
    if image is None:
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            "❌ Please upload an image",
            "",
            "",
            None
        )
    
    # Convert to PIL if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image).convert("RGB")
    else:
        image = image.convert("RGB")
    
    try:
        # Run the agentic workflow
        result = run_analysis(
            image=image,
            patient_id=current_patient.get("id", "Anonymous"),
            patient_context=f"Name: {current_patient.get('name')}, DOB: {current_patient.get('dob')}, Gender: {current_patient.get('gender')}"
        )
        
        result["patient_name"] = current_patient.get("name", "")
        result["patient_dob"] = current_patient.get("dob", "")
        result["patient_gender"] = current_patient.get("gender", "")
        result["patient_id"] = current_patient.get("id", "")
        
        # Generate enhanced report
        from agents.report_generator import generate_report
        report = generate_report(
            patient_name=result["patient_name"],
            patient_dob=result["patient_dob"],
            patient_gender=result["patient_gender"],
            classification=result.get("classification", "Unknown"),
            confidence=result.get("confidence", 0.0),
            clinical_advice=result.get("clinical_advice"),
            next_steps=result.get("next_steps"),
            severity=result.get("severity"),
            patient_id=result["patient_id"]
        )
        
        # Generate PDF
        pdf_path = generate_pdf_report(
            patient_name=result["patient_name"],
            patient_dob=result["patient_dob"],
            patient_id=result["patient_id"],
            classification=result.get("classification", "Unknown"),
            confidence=result.get("confidence", 0.0),
            clinical_advice=result.get("clinical_advice"),
            next_steps=result.get("next_steps")
        )
        
        # Workflow trace
        classification = result.get("classification", "Unknown")
        confidence = result.get("confidence", 0.0)
        
        trace = f"""
### Workflow Execution

| Step | Agent | Status |
|------|-------|--------|
| 1 | 🔬 Image Analyzer | ✅ Complete |
| 2 | 🩺 Clinical Advisor | {"✅ Complete" if result.get("clinical_complete") else "⏭️ Skipped"} |
| 3 | 📋 Report Generator | ✅ Complete |

**Classification**: {classification} ({confidence:.1%} confidence)
"""
        
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            "✅ Analysis complete!",
            report,
            trace,
            pdf_path
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            f"❌ Error: {str(e)}",
            "",
            "",
            None
        )


def start_new_analysis():
    """Reset and start a new analysis"""
    global current_patient
    current_patient = {}
    
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        "",
        "",
        "Not specified",
        None,
        "",
        "",
        None
    )


# ==================== Build Gradio UI ====================

custom_css = """
.gradio-container { max-width: 1000px !important; }
.step-header { 
    background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
    color: white;
    padding: 15px 20px;
    border-radius: 10px;
    margin-bottom: 20px;
}
/* Dark mode: force all HTML content to keep readable text on their light backgrounds */
.dark .ls-card {
    color: #1e293b !important;
}
.dark .ls-card * {
    color: inherit !important;
}
.dark .ls-card h1, .dark .ls-card h2,
.dark .ls-card h3, .dark .ls-card h4 {
    color: inherit !important;
}
.dark .ls-card code {
    color: #1e40af !important;
    background: rgba(0,0,0,0.06) !important;
}
/* Header stays white text */
.dark .ls-header, .dark .ls-header * {
    color: white !important;
}
/* Footer */
.dark .ls-footer {
    background: #1e293b !important;
}
.dark .ls-footer p {
    color: #94a3b8 !important;
}
.dark .ls-footer code {
    color: #60a5fa !important;
}
"""


def accept_disclaimer():
    """Hide disclaimer and show the main app"""
    return gr.update(visible=False), gr.update(visible=True)


with gr.Blocks(
    title="LeukemiaScope - AI Blood Cell Analysis",
    theme=gr.themes.Soft(),
    css=custom_css
) as demo:
    
    # ==================== DISCLAIMER POPUP ====================
    with gr.Group(visible=True) as disclaimer_section:
        gr.HTML("""
        <div class="ls-header" style="text-align: center; padding: 20px; background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%); color: white; border-radius: 12px; margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 32px; color: white;">🩸 LeukemiaScope</h1>
            <p style="margin: 10px 0 0 0; font-size: 16px; opacity: 0.9; color: white;">AI-Powered Blood Cell Analysis with Multi-Agent Workflow</p>
        </div>
        """)
        
        gr.HTML("""
        <div class="ls-card" style="background: #fef2f2; border: 2px solid #fecaca; border-radius: 12px; padding: 25px; margin-bottom: 20px; color: #7f1d1d;">
            <h2 style="margin-top: 0; color: #dc2626; text-align: center;">⚠️ Important Disclaimer</h2>
            <ul style="color: #7f1d1d; line-height: 2; font-size: 15px; padding-left: 20px;">
                <li>This tool is for <strong>research and educational purposes only</strong></li>
                <li>This is <strong>NOT a certified medical diagnostic tool</strong></li>
                <li>Results must be confirmed by qualified healthcare professionals</li>
                <li><strong>Do NOT</strong> make any treatment decisions based solely on this tool's output</li>
                <li>Always consult a hematologist or oncologist for definitive diagnosis</li>
            </ul>
        </div>
        """)
        
        gr.HTML("""
        <div class="ls-card" style="background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 12px; padding: 25px; margin-bottom: 20px; color: #1e3a5f;">
            <h2 style="margin-top: 0; color: #1e40af;">🤖 How This App Works</h2>
            <p style="color: #1e3a5f; line-height: 1.6;">LeukemiaScope uses a <strong>multi-agent AI workflow</strong> powered by <strong>LangGraph</strong> to analyze blood cell images:</p>
            
            <div style="display: flex; flex-direction: column; gap: 12px; margin: 15px 0;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div style="background: #3b82f6; color: white !important; width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; flex-shrink: 0;">1</div>
                    <div style="color: #1e3a5f;"><strong>🔬 Image Analyzer Agent</strong> — Fine-tuned <code>MedGemma 1.5 4B</code> with LoRA adapter analyzes your blood cell microscopy image and classifies it as <strong>Normal</strong> or <strong>Leukemia</strong>.</div>
                </div>
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div style="background: #3b82f6; color: white !important; width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; flex-shrink: 0;">2</div>
                    <div style="color: #1e3a5f;"><strong>🩺 Clinical Advisor Agent</strong> — If leukemia is detected, <code>Gemini 3 Flash Preview</code> provides clinical recommendations, next steps, and risk assessment.</div>
                </div>
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div style="background: #3b82f6; color: white !important; width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; flex-shrink: 0;">3</div>
                    <div style="color: #1e3a5f;"><strong>📋 Report Generator Agent</strong> — Generates a structured medical report with HTML view and downloadable PDF.</div>
                </div>
            </div>
        </div>
        """)
        
        gr.HTML("""
        <div class="ls-card" style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 12px; padding: 25px; margin-bottom: 20px; color: #14532d;">
            <h2 style="margin-top: 0; color: #166534;">📷 Image Requirements</h2>
            <p style="color: #14532d; line-height: 1.6;">For accurate results, please upload images that meet these criteria:</p>
            <ul style="color: #14532d; line-height: 2; font-size: 15px; padding-left: 20px;">
                <li><strong>Blood cell microscopy image</strong> — standard blood smear slides</li>
                <li><strong>Dark/black background</strong> — single cell or few cells visible</li>
                <li><strong>Stained samples preferred</strong> — Wright or Giemsa stain</li>
                <li><strong>Clear, focused image</strong> — avoid blurry or low-resolution images</li>
                <li><strong>RGB format</strong> — JPEG, PNG, or similar formats</li>
            </ul>
            <div style="background: #dcfce7; padding: 12px; border-radius: 8px; margin-top: 10px;">
                <p style="margin: 0; color: #166534; font-size: 14px;">
                    💡 <strong>Tip:</strong> The model was trained on ALL-IDB (Acute Lymphoblastic Leukemia Image Database) style images — 
                    single blood cell crops with dark backgrounds work best.
                </p>
            </div>
        </div>
        """)
        
        accept_btn = gr.Button(
            "✅ I Understand & Accept — Proceed to App", 
            variant="primary", 
            size="lg"
        )
    
    # ==================== MAIN APP (hidden until disclaimer accepted) ====================
    with gr.Group(visible=False) as main_app:
        
        # Header
        gr.HTML("""
        <div class="ls-header" style="text-align: center; padding: 20px; background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%); color: white; border-radius: 12px; margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 32px; color: white;">🩸 LeukemiaScope</h1>
            <p style="margin: 10px 0 0 0; font-size: 16px; opacity: 0.9; color: white;">AI-Powered Blood Cell Analysis with Multi-Agent Workflow</p>
        </div>
        """)
        
        # Progress indicator
        gr.HTML("""
        <div style="display: flex; justify-content: center; margin-bottom: 20px;">
            <div style="display: flex; align-items: center;">
                <div style="width: 35px; height: 35px; border-radius: 50%; background: #dc2626; color: white !important; display: flex; align-items: center; justify-content: center; font-weight: bold;">1</div>
                <div style="width: 80px; height: 3px; background: #e5e7eb;"></div>
                <div style="width: 35px; height: 35px; border-radius: 50%; background: #e5e7eb; color: #6b7280 !important; display: flex; align-items: center; justify-content: center; font-weight: bold;">2</div>
                <div style="width: 80px; height: 3px; background: #e5e7eb;"></div>
                <div style="width: 35px; height: 35px; border-radius: 50%; background: #e5e7eb; color: #6b7280 !important; display: flex; align-items: center; justify-content: center; font-weight: bold;">3</div>
            </div>
        </div>
        <div style="display: flex; justify-content: center; gap: 50px; margin-bottom: 30px; font-size: 14px; color: #9ca3af;">
            <span>Patient Info</span>
            <span>Image Upload</span>
            <span>Report</span>
        </div>
        """)
        
        status_msg = gr.Markdown("")
        
        # Step 1: Patient Information
        with gr.Group(visible=True) as step1:
            gr.Markdown("## 📋 Step 1: Patient Information")
            gr.Markdown("Please enter the patient details before proceeding with the analysis.")
            
            with gr.Row():
                with gr.Column():
                    patient_name = gr.Textbox(label="Full Name *", placeholder="Enter patient's full name", max_lines=1)
                    patient_dob = gr.Textbox(label="Date of Birth", placeholder="YYYY-MM-DD", max_lines=1)
                    patient_gender = gr.Dropdown(label="Gender", choices=["Not specified", "Male", "Female", "Other"], value="Not specified")
            
            next_btn_1 = gr.Button("Continue to Image Upload →", variant="primary", size="lg")
        
        # Step 2: Image Upload
        with gr.Group(visible=False) as step2:
            gr.Markdown("## 📷 Step 2: Upload Blood Cell Image")
            gr.Markdown("Upload a microscopy image of blood cells for analysis.")
            
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(label="Blood Cell Image", type="pil", height=350)
                    with gr.Row():
                        back_btn = gr.Button("← Back", size="lg")
                        analyze_btn = gr.Button("🔬 Analyze Image", variant="primary", size="lg")
        
        # Step 3: Results
        with gr.Group(visible=False) as step3:
            gr.Markdown("## 📊 Step 3: Analysis Results")
            
            with gr.Row():
                with gr.Column(scale=2):
                    report_output = gr.HTML(label="Medical Report")
                with gr.Column(scale=1):
                    trace_output = gr.Markdown(label="Workflow Trace")
                    gr.Markdown("### 📥 Download Report")
                    pdf_download = gr.File(label="PDF Report", interactive=False)
                    new_analysis_btn = gr.Button("🔄 New Analysis", variant="secondary", size="lg")
    
    # ==================== Event Handlers ====================
    
    # Disclaimer accept
    accept_btn.click(accept_disclaimer, [], [disclaimer_section, main_app])
    
    # Step 1 -> Step 2
    next_btn_1.click(save_patient_info, [patient_name, patient_dob, patient_gender], [step1, step2, step3, status_msg])
    back_btn.click(go_back_to_step1, [], [step1, step2, step3])
    analyze_btn.click(analyze_image_workflow, [image_input], [step2, step3, status_msg, report_output, trace_output, pdf_download])
    new_analysis_btn.click(start_new_analysis, [], [step1, step2, step3, patient_name, patient_dob, patient_gender, image_input, report_output, trace_output, pdf_download])
    
    # Footer
    gr.HTML("""
    <div class="ls-footer" style="margin-top: 30px; padding: 20px; background: #f8fafc; border-radius: 10px; text-align: center;">
        <p style="margin: 0; color: #64748b; font-size: 14px;">
            Powered by <strong style="color: #64748b;">LangGraph</strong> Multi-Agent Workflow | 
            Model: <code style="color: #3b82f6;">chaudhrysuleman/medgemma-1.5-4b-it-leukemia-lora</code>
        </p>
        <p style="margin: 10px 0 0 0; color: #94a3b8; font-size: 12px;">
            Built for the MedGemma Impact Challenge 2026 | By Chaudhry Muhammad Suleman &amp; Muhammad Idnan
        </p>
    </div>
    """)


# Pre-load model at startup
print("=" * 60)
print("🩸 LeukemiaScope - Agentic AI Workflow")
print("=" * 60)
print("📥 Pre-loading MedGemma model...")
predictor = get_predictor()
predictor.load()
print("✅ Model loaded! Launching app...")

# Launch for HuggingFace Spaces
demo.launch(server_name="0.0.0.0", server_port=7860)
