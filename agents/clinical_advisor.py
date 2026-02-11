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
                f"with {confidence:.1%} confidence. No abnormal lymphoblast morphology was detected.\n\n"
                "**Clinical Interpretation:** The submitted peripheral blood smear image does not show "
                "features consistent with Acute Lymphoblastic Leukemia (ALL). The cell morphology appears "
                "within normal limits — round nucleus with mature chromatin pattern, adequate cytoplasm, "
                "and no blast-like characteristics.\n\n"
                "**Recommendations:**\n"
                "• No immediate haematological intervention is indicated based on this screening.\n"
                "• If the patient presents with clinical symptoms (persistent fatigue, unexplained bruising, "
                "recurrent infections, bone pain, lymphadenopathy), a full CBC with manual differential "
                "should still be performed regardless of this AI result.\n"
                "• Routine follow-up as per standard clinical guidelines.\n\n"
                "**Important:** A normal AI screening result does not constitute a definitive diagnosis. "
                "Clinical correlation with patient symptoms, physical examination, and laboratory findings "
                "is always required."
            ),
            "next_steps": [
                "Continue routine monitoring as clinically indicated",
                "Perform CBC with manual differential if patient has symptoms",
                "Document screening result in patient records",
                "Schedule routine follow-up per clinical guidelines"
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
        ("system", """You are a senior clinical haematologist AI advisor providing detailed clinical 
recommendations based on an AI blood cell screening result. You must provide thorough, 
structured medical guidance.

Use this clinical knowledge base for reference:
{knowledge}

IMPORTANT GUIDELINES:
- Be thorough and detailed — this report will be included in a medical screening document
- Use professional medical terminology with explanations
- Structure your response with clear sections using markdown headers (##)
- Always emphasise this is AI-assisted screening, NOT a definitive diagnosis
- Include specific diagnostic tests with clinical rationale for each
- Provide actionable next steps with priority levels
- Address urgency and timeline for follow-up
- Consider the patient context if provided"""),
        ("human", """Blood cell AI screening result:
- Classification: {classification}
- AI Confidence: {confidence:.1%}
{patient_info}

Please provide a comprehensive clinical advisory report with the following sections:

## Clinical Interpretation
Explain what the AI finding means clinically. Describe the morphological features that 
suggest Acute Lymphoblastic Leukemia and the clinical significance.

## Severity Assessment
Rate the urgency (High/Critical) and explain why immediate follow-up is necessary.

## Recommended Diagnostic Workup
List specific diagnostic tests in order of priority, with brief rationale for each:
- Complete Blood Count (CBC) with manual differential
- Peripheral blood smear review
- Bone marrow aspiration and biopsy
- Flow cytometry / immunophenotyping
- Cytogenetic analysis
- Molecular testing
- Any additional tests as indicated

## Treatment Considerations
Briefly outline the treatment pathway if diagnosis is confirmed, including:
- Initial management priorities
- Standard treatment phases
- Targeted therapies if applicable

## Patient Communication
Key points for communicating this screening result to the patient or their family, 
emphasising the need for confirmatory testing and avoiding premature alarm.

## Timeline & Urgency
Provide specific timeline recommendations for diagnostic workup and specialist referral.""")
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
        "Bone marrow aspiration and biopsy for definitive diagnosis",
        "Flow cytometry / immunophenotyping (CD19, CD10, CD22, TdT)",
        "Cytogenetic analysis (karyotype + FISH) for risk stratification",
        "Molecular testing (BCR-ABL1, MLL rearrangements)",
        "Refer to haematologist/oncologist within 24-48 hours",
        "Baseline metabolic panel and coagulation studies"
    ]
    
    # Try parsing numbered items from the "Diagnostic Workup" section
    steps = []
    in_workup = False
    for line in advice_text.split('\n'):
        stripped = line.strip()
        if 'diagnostic workup' in stripped.lower() or 'recommended diagnostic' in stripped.lower():
            in_workup = True
            continue
        if in_workup and stripped.startswith('#'):
            break  # Next section
        if in_workup and (stripped.startswith('-') or stripped.startswith('*') or 
                          (len(stripped) > 2 and stripped[0].isdigit() and stripped[1] in '.)')):
            clean = stripped.lstrip('-*0123456789.) ').strip()
            if clean and len(clean) > 5:
                steps.append(clean)
    
    return steps if len(steps) >= 3 else default_steps


def _generate_fallback_advice(confidence: float, patient_context: Optional[str] = None) -> dict:
    """Generate comprehensive clinical advice without LLM."""
    severity = "Critical" if confidence > 0.85 else "High"
    
    patient_note = ""
    if patient_context:
        patient_note = f"\n**Patient Context:** {patient_context}\n"
    
    recommendations = f"""## Clinical Interpretation

The AI screening model has identified morphological features consistent with **Acute Lymphoblastic Leukemia (ALL)** with **{confidence:.1%} confidence**. The analysed blood cell image shows characteristics suggestive of abnormal lymphoblast proliferation, including potential high nuclear-to-cytoplasm ratio and immature chromatin pattern.
{patient_note}
**Important:** This is an AI-assisted screening result, NOT a confirmed diagnosis. Definitive diagnosis requires comprehensive laboratory evaluation and expert haematopathological review.

## Severity Assessment

**Severity Level: {severity}**

This finding requires **urgent clinical follow-up**. Acute Lymphoblastic Leukemia is a rapidly progressive malignancy. Early detection and prompt initiation of treatment are critical factors in patient outcomes. The AI model's confidence level of {confidence:.1%} warrants immediate escalation to the diagnostic pathway.

## Recommended Diagnostic Workup

The following tests should be performed in order of priority:

1. **Complete Blood Count (CBC) with Manual Differential** — Assess total WBC count, identify blast percentage, evaluate for anaemia and thrombocytopenia
2. **Peripheral Blood Smear Review** — Expert morphological assessment by haematopathologist to characterise blast cell features
3. **Bone Marrow Aspiration & Biopsy** — Gold standard for ALL diagnosis; ≥20% blasts confirms diagnosis (WHO criteria)
4. **Flow Cytometry / Immunophenotyping** — Distinguish B-ALL vs T-ALL; identify CD markers (CD19, CD10, CD22, TdT)
5. **Cytogenetic Analysis (Karyotype + FISH)** — Identify prognostically significant translocations (t(12;21), t(9;22) Philadelphia chromosome)
6. **Molecular Testing (PCR)** — BCR-ABL1 fusion, MLL rearrangements, iAMP21
7. **Lumbar Puncture** — Assess CNS involvement (after bone marrow confirmation)
8. **Baseline Metabolic Panel & Coagulation Studies** — Assess organ function and tumour lysis risk

## Treatment Considerations

If ALL is confirmed, the standard treatment pathway includes:
- **Remission Induction** (4-6 weeks): Vincristine, corticosteroids, L-asparaginase ± anthracycline
- **Consolidation / Intensification**: High-dose methotrexate, cytarabine
- **CNS Prophylaxis**: Intrathecal methotrexate
- **Maintenance Therapy** (2-3 years): Daily 6-mercaptopurine, weekly methotrexate
- **Targeted Therapy** (if Ph+): Tyrosine kinase inhibitors (imatinib/dasatinib)

## Patient Communication

- Explain that the AI screening detected cells that require further evaluation
- Emphasise that additional confirmatory tests are needed before any diagnosis
- Avoid using definitive language — refer to it as a "screening finding" requiring investigation
- Reassure that modern treatment protocols for ALL are highly effective, especially in children (85-90% cure rate)
- Provide emotional support resources and clear next steps

## Timeline & Urgency

- **Within 24 hours**: CBC with differential and peripheral blood smear review
- **Within 48-72 hours**: Haematologist/oncologist referral and bone marrow biopsy scheduling
- **Within 1 week**: Complete diagnostic workup including flow cytometry and cytogenetics
- **Ongoing**: Results review and treatment planning conference"""
    
    return {
        "recommendations": recommendations,
        "next_steps": [
            "Urgent: Complete Blood Count (CBC) with manual differential",
            "Peripheral blood smear review by haematopathologist",
            "Bone marrow aspiration and biopsy for definitive diagnosis",
            "Flow cytometry / immunophenotyping (CD19, CD10, CD22, TdT)",
            "Cytogenetic analysis (karyotype + FISH) for risk stratification",
            "Molecular testing (BCR-ABL1, MLL rearrangements)",
            "Refer to haematologist/oncologist within 24-48 hours",
            "Baseline metabolic panel and coagulation studies"
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
