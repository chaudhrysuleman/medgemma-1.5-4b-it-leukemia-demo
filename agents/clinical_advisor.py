"""
Clinical Advisor Agent
Provides detailed clinical recommendations when leukemia is detected
"""

import os
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


# Clinical knowledge base
LEUKEMIA_KNOWLEDGE = """
## Acute Lymphoblastic Leukemia (ALL) Clinical Information

### Overview
Acute Lymphoblastic Leukemia (ALL) is a malignant neoplasm of the blood and bone marrow
characterised by the clonal proliferation of immature lymphoid precursors (lymphoblasts).
ALL is the most common childhood malignancy, accounting for ~25% of all pediatric cancers,
but also occurs in adults with a second incidence peak after age 50.

### Key Clinical Features
- Abnormal lymphoblast cells infiltrating blood and bone marrow
- Rapid progression if untreated; median survival without treatment is weeks to months
- 5-year survival rate: 85-90% in children, 35-40% in adults with current protocols

### Morphological Indicators (Peripheral Blood Smear)
- Blast cells with high nuclear-to-cytoplasm ratio
- Fine, dispersed chromatin pattern
- Inconspicuous to prominent nucleoli
- Scant, agranular, basophilic cytoplasm
- Possible Auer rods (more common in AML, but must be excluded)

### Recommended Diagnostic Workup for Positive Screening
1. **Complete Blood Count (CBC) with Manual Differential**: Assess WBC count, blast percentage, anaemia, thrombocytopenia
2. **Peripheral Blood Smear Review**: Morphological assessment by haematopathologist
3. **Bone Marrow Aspiration & Biopsy**: Gold standard; ≥20% blasts confirms diagnosis (WHO) or ≥25% (COG)
4. **Flow Cytometry / Immunophenotyping**: Distinguish B-ALL vs T-ALL; identify CD markers (CD19, CD10, CD22, TdT)
5. **Cytogenetic Analysis (Karyotype + FISH)**: Identify translocations (e.g., t(12;21), t(9;22) Philadelphia chromosome)
6. **Molecular Testing**: PCR for BCR-ABL1, MLL rearrangements, iAMP21
7. **Lumbar Puncture**: Assess CNS involvement
8. **Metabolic Panel & Coagulation Studies**: Baseline organ function, tumour lysis risk

### Risk Stratification (NCI Criteria)
- **Standard Risk**: Age 1-9.99 years, WBC <50,000/μL, favourable cytogenetics
- **High Risk**: Age ≥10 or <1 year, WBC ≥50,000/μL, unfavourable cytogenetics (Ph+, MLL)
- **Very High Risk**: Induction failure, MRD-positive post-induction, hypodiploidy (<44 chromosomes)

### Treatment Overview
- **Remission Induction** (4-6 weeks): Vincristine, corticosteroids, L-asparaginase ± anthracycline
- **Consolidation / Intensification**: High-dose methotrexate, cytarabine, cyclophosphamide
- **CNS Prophylaxis**: Intrathecal methotrexate ± cranial radiation (selected cases)
- **Maintenance Therapy** (2-3 years): Daily 6-mercaptopurine, weekly methotrexate
- **Targeted Therapy**: Imatinib/dasatinib for Ph+ ALL; blinatumomab, inotuzumab for relapsed/refractory

### Prognosis Factors
- Favourable: Age 1-9, standard risk, t(12;21)/ETV6-RUNX1, hyperdiploidy (51-65), rapid early response
- Unfavourable: Ph+, MLL rearrangement, hypodiploidy, CNS involvement, induction failure
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
    Generate clinical recommendations based on blood cell classification.
    
    Args:
        classification: "Leukemia" or "Normal"
        confidence: Model confidence score (0.0 - 1.0)
        patient_context: Optional additional patient info (name, age, gender)
        
    Returns:
        dict with recommendations, next_steps, severity, requires_urgent_action
    """
    # ---- Normal result ----
    if classification != "Leukemia":
        return {
            "recommendations": (
                "The AI screening model classified this blood cell image as **Normal** "
                f"({confidence:.1%} confidence). No abnormal lymphoblast morphology was detected.\n\n"
                "## Clinical Summary\n"
                "The cell morphology appears within normal limits with no blast-like characteristics. "
                "This is a screening result, not a clinical diagnosis.\n\n"
                "## Recommended Actions\n"
                "• No immediate haematological intervention in the absence of symptoms.\n"
                "• Routine clinical follow-up as indicated."
            ),
            "next_steps": [
                "Routine monitoring",
                "CBC if symptoms present"
            ],
            "severity": "Low",
            "requires_urgent_action": False
        }
    
    # ---- Leukemia detected ----
    llm = get_clinical_llm()
    
    if llm is None:
        # Comprehensive fallback without LLM
        return _generate_fallback_advice(confidence, patient_context)
    
    # Use LLM for detailed, personalized clinical advice
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a senior clinical haematologist AI advisor providing a CONCISE clinical summary based on an AI blood cell screening result. 
        
Use this clinical knowledge base for reference:
{knowledge}

IMPORTANT GUIDELINES:
- Be extremely CONCISE and direct.
- Limit response to 2 key sections: "Clinical Summary" and "Recommended Actions".
- Total response should be under 200 words.
- Emphasise this is AI-assisted screening, NOT a definitive diagnosis.
- Provide a bulleted list of prioritized next steps.
- Do NOT include generic treatment protocols or patient communication advice unless critical."""),
        ("human", """Blood cell AI screening result:
- Classification: {classification}
- AI Confidence: {confidence:.1%}
{patient_info}

Please provide a concise clinical report:

## Clinical Summary
Briefly explain the finding (morphology & significance) in 2-3 sentences. Identify if this is a high-risk finding.

## Recommended Actions
List the top 3-5 most critical next steps (e.g., CBC, Smear Review, Referral) in order of urgency.
""")
    ])
    
    patient_info = f"- Patient Context: {patient_context}" if patient_context else "- Patient Context: Not provided"
    
    try:
        response = llm.invoke(
            prompt.format_messages(
                knowledge=LEUKEMIA_KNOWLEDGE,
                classification=classification,
                confidence=confidence,
                patient_info=patient_info
            )
        )
        
        # Extract text
        content = response.content
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and 'text' in part:
                    text_parts.append(part['text'])
                elif isinstance(part, str):
                    text_parts.append(part)
            advice_text = "\n".join(text_parts)
        else:
            advice_text = str(content)
        
        # Extract next steps from LLM response
        next_steps = _extract_next_steps(advice_text)
        
        # Determine severity from response
        severity = "Critical" if confidence > 0.85 else "High"
        
        return {
            "recommendations": advice_text,
            "next_steps": next_steps,
            "severity": severity,
            "requires_urgent_action": True,
            "knowledge_base": LEUKEMIA_KNOWLEDGE
        }
        
    except Exception as e:
        print(f"⚠️ Clinical LLM error: {e}, using fallback")
        return _generate_fallback_advice(confidence, patient_context)


def _extract_next_steps(advice_text: str) -> list:
    """Extract actionable next steps from the LLM clinical advice text."""
    default_steps = [
        "Urgent: Complete Blood Count (CBC) with manual differential",
        "Peripheral blood smear review by haematopathologist",
        "Bone marrow aspiration for definitive diagnosis",
        "Flow cytometry / immunophenotyping",
        "Refer to haematologist within 24-48 hours"
    ]
    
    # Try parsing numbered/bulleted items from the text
    steps = []
    in_actions = False
    for line in advice_text.split('\n'):
        stripped = line.strip()
        if 'recommended actions' in stripped.lower() or 'next steps' in stripped.lower():
            in_actions = True
            continue
        if in_actions and stripped.startswith('#'):
            break  # Next section
        if in_actions and (stripped.startswith('-') or stripped.startswith('*') or 
                          (len(stripped) > 2 and stripped[0].isdigit() and stripped[1] in '.)')):
            clean = stripped.lstrip('-*0123456789.) ').strip()
            if clean and len(clean) > 5:
                steps.append(clean)
    
    return steps if len(steps) >= 3 else default_steps


def _generate_fallback_advice(confidence: float, patient_context: Optional[str] = None) -> dict:
    """Generate concise clinical advice without LLM."""
    severity = "Critical" if confidence > 0.85 else "High"
    
    patient_note = ""
    if patient_context:
        patient_note = f"\n**Patient Context:** {patient_context}\n"
    
    recommendations = f"""## Clinical Summary
The AI model identified features consistent with **Acute Lymphoblastic Leukemia (ALL)** ({confidence:.1%} confidence). Visual indicators suggest abnormal blast proliferation. **This is a screening result, not a confirmed diagnosis.** Immediate haematological evaluation is required to confirm.
{patient_note}

## Recommended Actions
1. **Urgent CBC with Manual Differential** to assess blast count and cytopenias.
2. **Peripheral Blood Smear Review** by a haematopathologist.
3. **Bone Marrow Biopsy & Flow Cytometry** for definitive diagnosis.
4. **Referral to Haematology/Oncology** within 24-48 hours."""
    
    return {
        "recommendations": recommendations,
        "next_steps": [
            "Urgent: Complete Blood Count (CBC) with manual differential",
            "Peripheral blood smear review by haematopathologist",
            "Bone marrow aspiration/biopsy & Flow Cytometry",
            "Refer to haematologist within 24-48 hours"
        ],
        "severity": severity,
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
