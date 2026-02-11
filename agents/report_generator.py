"""
Report Generator Agent - Enhanced Version
Generates structured medical reports with PDF export
"""

import os
from datetime import datetime
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


def get_report_llm():
    """Get the report generator LLM"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    
    return ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        google_api_key=api_key,
        temperature=0.2
    )


def calculate_age(dob: str) -> str:
    """Calculate age from date of birth string"""
    try:
        if not dob:
            return "Unknown"
        birth = datetime.strptime(dob, "%Y-%m-%d")
        today = datetime.today()
        age = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
        return f"{age} years"
    except:
        return "Unknown"


def generate_report(
    patient_name: str,
    patient_dob: str,
    patient_gender: str,
    classification: str,
    confidence: float,
    clinical_advice: Optional[str] = None,
    next_steps: Optional[list] = None,
    severity: Optional[str] = None,
    patient_id: Optional[str] = None
) -> str:
    """
    Generate an enhanced structured medical report
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    patient_id = patient_id or f"LS-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    age = calculate_age(patient_dob)
    
    # Status formatting
    if classification == "Normal":
        status_emoji = "✅"
        status_text = "NORMAL"
        alert_class = "normal"
        alert_color = "#22c55e"
    elif classification == "Leukemia":
        status_emoji = "⚠️"
        status_text = "LEUKEMIA DETECTED"
        alert_class = "leukemia"
        alert_color = "#ef4444"
    else:
        status_emoji = "🔍"
        status_text = "UNCERTAIN"
        alert_class = "uncertain"
        alert_color = "#eab308"
    
    # Build report with HTML styling for Gradio
    report = f"""
<div class="ls-card" style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 800px; margin: 0 auto; color: #1e293b;">

<!-- Header -->
<div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%); color: white; border-radius: 12px 12px 0 0;">
    <h1 style="margin: 0; font-size: 28px;">🩸 LeukemiaScope</h1>
    <p style="margin: 5px 0 0 0; opacity: 0.9;">AI Blood Cell Analysis Report</p>
</div>

<!-- Patient Information Card -->
<div style="background: #f8fafc; padding: 20px; border-left: 4px solid #3b82f6;">
    <h3 style="margin-top: 0; color: #1e40af;">📋 Patient Information</h3>
    <table style="width: 100%; border-collapse: collapse;">
        <tr>
            <td style="padding: 8px 0; color: #64748b; width: 140px;"><strong>Name:</strong></td>
            <td style="padding: 8px 0;">{patient_name or 'Not provided'}</td>
            <td style="padding: 8px 0; color: #64748b; width: 140px;"><strong>Patient ID:</strong></td>
            <td style="padding: 8px 0;">{patient_id}</td>
        </tr>
        <tr>
            <td style="padding: 8px 0; color: #64748b;"><strong>Date of Birth:</strong></td>
            <td style="padding: 8px 0;">{patient_dob or 'Not provided'}</td>
            <td style="padding: 8px 0; color: #64748b;"><strong>Age:</strong></td>
            <td style="padding: 8px 0;">{age}</td>
        </tr>
        <tr>
            <td style="padding: 8px 0; color: #64748b;"><strong>Gender:</strong></td>
            <td style="padding: 8px 0;">{patient_gender or 'Not specified'}</td>
            <td style="padding: 8px 0; color: #64748b;"><strong>Report Date:</strong></td>
            <td style="padding: 8px 0;">{timestamp}</td>
        </tr>
    </table>
</div>

<!-- Classification Result -->
<div style="background: {alert_color}; color: white; padding: 25px; text-align: center; margin: 20px 0; border-radius: 8px;">
    <h2 style="margin: 0; font-size: 28px;">{status_emoji} {status_text}</h2>
    <div style="margin-top: 15px; font-size: 18px;">
        <span style="background: rgba(255,255,255,0.2); padding: 8px 20px; border-radius: 20px;">
            Confidence: <strong>{confidence:.1%}</strong>
        </span>
    </div>
</div>

<!-- Analysis Details -->
<div style="background: white; padding: 20px; border: 1px solid #e2e8f0; border-radius: 8px; margin-bottom: 20px;">
    <h3 style="margin-top: 0; color: #1e293b;">🔬 Analysis Details</h3>
    <table style="width: 100%; border-collapse: collapse;">
        <tr style="border-bottom: 1px solid #e2e8f0;">
            <td style="padding: 12px 0; color: #64748b;">Analysis Method</td>
            <td style="padding: 12px 0; text-align: right;"><strong>MedGemma 1.5 4B (Fine-tuned LoRA)</strong></td>
        </tr>
        <tr style="border-bottom: 1px solid #e2e8f0;">
            <td style="padding: 12px 0; color: #64748b;">Model ID</td>
            <td style="padding: 12px 0; text-align: right;"><code>chaudhrysuleman/medgemma-1.5-4b-it-leukemia-lora</code></td>
        </tr>
        <tr style="border-bottom: 1px solid #e2e8f0;">
            <td style="padding: 12px 0; color: #64748b;">Model Accuracy</td>
            <td style="padding: 12px 0; text-align: right;"><strong>78.15%</strong></td>
        </tr>
        <tr>
            <td style="padding: 12px 0; color: #64748b;">Leukemia Recall</td>
            <td style="padding: 12px 0; text-align: right;"><strong>83.10%</strong> (optimized for sensitivity)</td>
        </tr>
    </table>
</div>
"""

    # Clinical Advice Section (if leukemia detected)
    if classification == "Leukemia" and clinical_advice:
        # Sanitize clinical_advice — may be string, list, or dict
        if isinstance(clinical_advice, list):
            parts = []
            for item in clinical_advice:
                if isinstance(item, dict) and 'text' in item:
                    parts.append(item['text'])
                elif isinstance(item, str):
                    parts.append(item)
            advice_text = "\n".join(parts)
        elif isinstance(clinical_advice, dict):
            advice_text = clinical_advice.get('text', str(clinical_advice))
        else:
            advice_text = str(clinical_advice)
        
        # Convert markdown to basic HTML
        advice_html = advice_text.replace('\n\n', '</p><p style="color: #7f1d1d; line-height: 1.6;">')
        advice_html = advice_html.replace('\n', '<br>')
        
        report += f"""
<!-- Clinical Recommendations -->
<div style="background: #fef2f2; padding: 20px; border: 1px solid #fecaca; border-radius: 8px; margin-bottom: 20px;">
    <h3 style="margin-top: 0; color: #dc2626;">🩺 Clinical Recommendations</h3>
    <p style="color: #7f1d1d; line-height: 1.6;">{advice_html}</p>
</div>
"""

    # Next Steps
    if next_steps:
        steps_html = ""
        for i, step in enumerate(next_steps, 1):
            clean_step = step.lstrip("0123456789. ")
            steps_html += f"""
        <div style="display: flex; align-items: flex-start; margin-bottom: 12px;">
            <div style="background: #3b82f6; color: white; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; flex-shrink: 0;">{i}</div>
            <div style="margin-left: 12px; padding-top: 4px;">{clean_step}</div>
        </div>
"""
        report += f"""
<!-- Next Steps -->
<div style="background: #eff6ff; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
    <h3 style="margin-top: 0; color: #1e40af;">📝 Recommended Next Steps</h3>
    {steps_html}
</div>
"""

    # Disclaimer
    report += """
<!-- Disclaimer -->
<div style="background: #fefce8; padding: 20px; border: 1px solid #fef08a; border-radius: 8px; margin-bottom: 20px;">
    <h3 style="margin-top: 0; color: #a16207;">⚠️ Important Disclaimer</h3>
    <ul style="color: #854d0e; margin: 0; padding-left: 20px; line-height: 1.8;">
        <li>This report is generated by an AI screening tool for <strong>research and educational purposes only</strong></li>
        <li>This is <strong>NOT a medical diagnosis</strong></li>
        <li>Results must be confirmed by qualified healthcare professionals</li>
        <li>Do not make treatment decisions based solely on this report</li>
        <li>Always consult a hematologist or oncologist for definitive diagnosis</li>
    </ul>
</div>

<!-- Footer -->
<div style="text-align: center; padding: 20px; color: #64748b; font-size: 12px;">
    <p>Report generated by <strong>LeukemiaScope</strong> - MedGemma Impact Challenge 2026</p>
    <p>By Chaudhry Muhammad Suleman & Muhammad Idnan</p>
</div>

</div>
"""
    
    return report


def report_generator_node(state: dict) -> dict:
    """
    LangGraph node for report generation
    """
    report = generate_report(
        patient_name=state.get("patient_name", ""),
        patient_dob=state.get("patient_dob", ""),
        patient_gender=state.get("patient_gender", ""),
        classification=state.get("classification", "Unknown"),
        confidence=state.get("confidence", 0.0),
        clinical_advice=state.get("clinical_advice"),
        next_steps=state.get("next_steps"),
        severity=state.get("severity"),
        patient_id=state.get("patient_id")
    )
    
    return {
        **state,
        "report": report,
        "report_complete": True
    }
