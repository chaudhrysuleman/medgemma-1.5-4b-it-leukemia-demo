"""
Clinical Advisor Agent
Provides clinical recommendations when leukemia is detected
"""

import os
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate


# Clinical knowledge base
LEUKEMIA_KNOWLEDGE = """
## Acute Lymphoblastic Leukemia (ALL) Clinical Information

### Overview
Acute Lymphoblastic Leukemia (ALL) is a cancer of the blood and bone marrow that affects 
white blood cells called lymphocytes. ALL is the most common type of cancer in children.

### Key Clinical Features
- Abnormal lymphoblast cells in blood/bone marrow
- Rapid progression if untreated
- 5-year survival rate: 85-90% with proper treatment

### Recommended Next Steps for Positive Screening
1. **Confirm Diagnosis**: Complete Blood Count (CBC) with differential
2. **Bone Marrow Biopsy**: Gold standard for ALL diagnosis
3. **Flow Cytometry**: Immunophenotyping of blast cells
4. **Cytogenetic Testing**: Chromosome analysis for prognosis
5. **Refer to Hematologist/Oncologist**: Specialized care required

### Risk Stratification
- Standard Risk: Age 1-9, WBC <50,000/μL
- High Risk: Age <1 or >10, WBC >50,000/μL

### Treatment Overview
- Induction chemotherapy
- Consolidation therapy
- Maintenance therapy (2-3 years)
- CNS prophylaxis
"""


def get_clinical_llm():
    """Get the clinical advisor LLM"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    
    return ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        google_api_key=api_key,
        temperature=0.3
    )


def generate_clinical_advice(
    classification: str,
    confidence: float,
    patient_context: Optional[str] = None
) -> dict:
    """
    Generate clinical recommendations for leukemia detection
    
    Args:
        classification: "Leukemia" or "Normal"
        confidence: Model confidence score
        patient_context: Optional additional patient info
        
    Returns:
        dict with recommendations, next_steps, severity
    """
    if classification != "Leukemia":
        return {
            "recommendations": "No immediate clinical action required for normal cells.",
            "next_steps": ["Regular monitoring", "Follow-up if symptoms develop"],
            "severity": "Low",
            "requires_urgent_action": False
        }
    
    llm = get_clinical_llm()
    
    if llm is None:
        # Fallback without LLM
        return {
            "recommendations": (
                "Leukemia blast cells detected. Immediate referral to hematologist recommended. "
                "Confirm diagnosis with CBC and bone marrow biopsy."
            ),
            "next_steps": [
                "1. Complete Blood Count (CBC) with differential",
                "2. Bone marrow biopsy for definitive diagnosis",
                "3. Refer to hematologist/oncologist",
                "4. Flow cytometry for cell typing",
                "5. Genetic testing for prognosis"
            ],
            "severity": "High",
            "requires_urgent_action": True,
            "knowledge_base": LEUKEMIA_KNOWLEDGE
        }
    
    # Use LLM for personalized advice
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a clinical advisor AI assistant. Based on the blood cell analysis 
showing potential leukemia, provide clinical recommendations.

Use this knowledge base:
{knowledge}

Be professional, accurate, and emphasize that this is AI screening, not diagnosis."""),
        ("human", """Blood cell analysis result:
- Classification: {classification}
- Confidence: {confidence:.1%}
{patient_info}

Provide:
1. Clinical interpretation
2. Recommended next steps (numbered list)
3. Urgency assessment
4. Key points for the patient/clinician""")
    ])
    
    patient_info = f"- Additional context: {patient_context}" if patient_context else ""
    
    response = llm.invoke(
        prompt.format_messages(
            knowledge=LEUKEMIA_KNOWLEDGE,
            classification=classification,
            confidence=confidence,
            patient_info=patient_info
        )
    )
    
    return {
        "recommendations": response.content,
        "next_steps": [
            "Complete Blood Count (CBC)",
            "Bone marrow biopsy",
            "Hematologist referral",
            "Flow cytometry",
            "Genetic testing"
        ],
        "severity": "High",
        "requires_urgent_action": True,
        "knowledge_base": LEUKEMIA_KNOWLEDGE
    }


def clinical_advisor_node(state: dict) -> dict:
    """
    LangGraph node for clinical advice
    
    Args:
        state: Graph state with classification results
        
    Returns:
        Updated state with clinical recommendations
    """
    classification = state.get("classification", "Unknown")
    confidence = state.get("confidence", 0.0)
    patient_context = state.get("patient_context")
    
    advice = generate_clinical_advice(
        classification=classification,
        confidence=confidence,
        patient_context=patient_context
    )
    
    return {
        **state,
        "clinical_advice": advice["recommendations"],
        "next_steps": advice["next_steps"],
        "severity": advice["severity"],
        "requires_urgent_action": advice["requires_urgent_action"],
        "clinical_complete": True
    }
