"""
LeukemiaScope Agentic Workflow
LangGraph-based multi-agent system for blood cell analysis
"""

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from PIL import Image

# Import agent nodes
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.image_analyzer import image_analyzer_node
from agents.clinical_advisor import clinical_advisor_node
from agents.report_generator import report_generator_node


class WorkflowState(TypedDict):
    """State shared across all agents in the workflow"""
    # Input
    image: Image.Image
    patient_id: str
    patient_context: str
    
    # Image Analysis Results
    classification: str
    confidence: float
    is_leukemia: bool
    raw_response: str
    analysis_complete: bool
    
    # Clinical Advice
    clinical_advice: str
    next_steps: list
    severity: str
    requires_urgent_action: bool
    clinical_complete: bool
    
    # Report
    report: str
    report_complete: bool
    
    # Errors
    error: str


def should_consult_clinical_advisor(state: WorkflowState) -> str:
    """
    Conditional edge: Route to Clinical Advisor only if leukemia is detected
    
    Args:
        state: Current workflow state
        
    Returns:
        "clinical_advisor" if leukemia detected, else "report_generator"
    """
    if state.get("is_leukemia", False):
        return "clinical_advisor"
    return "report_generator"


def create_workflow() -> StateGraph:
    """
    Create the LangGraph workflow for blood cell analysis
    
    Workflow:
    1. Image Analyzer (MedGemma) - Always runs first
    2. Clinical Advisor - Only if leukemia detected
    3. Report Generator - Always runs, generates final report
    """
    # Create the graph
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("image_analyzer", image_analyzer_node)
    workflow.add_node("clinical_advisor", clinical_advisor_node)
    workflow.add_node("report_generator", report_generator_node)
    
    # Set entry point
    workflow.set_entry_point("image_analyzer")
    
    # Add conditional edge from image_analyzer
    workflow.add_conditional_edges(
        "image_analyzer",
        should_consult_clinical_advisor,
        {
            "clinical_advisor": "clinical_advisor",
            "report_generator": "report_generator"
        }
    )
    
    # Clinical advisor always leads to report generator
    workflow.add_edge("clinical_advisor", "report_generator")
    
    # Report generator ends the workflow
    workflow.add_edge("report_generator", END)
    
    return workflow


def compile_workflow():
    """Compile the workflow into an executable graph"""
    workflow = create_workflow()
    return workflow.compile()


# Global compiled workflow
_app = None


def get_app():
    """Get or create the compiled workflow app"""
    global _app
    if _app is None:
        _app = compile_workflow()
    return _app


def run_analysis(image: Image.Image, patient_id: str = "Anonymous", patient_context: str = "") -> dict:
    """
    Run the complete analysis workflow
    
    Args:
        image: PIL Image of blood cell
        patient_id: Optional patient identifier
        patient_context: Optional context about patient
        
    Returns:
        Final state with all results
    """
    app = get_app()
    
    # Initial state
    initial_state = {
        "image": image,
        "patient_id": patient_id,
        "patient_context": patient_context,
        "classification": "",
        "confidence": 0.0,
        "is_leukemia": False,
        "raw_response": "",
        "analysis_complete": False,
        "clinical_advice": "",
        "next_steps": [],
        "severity": "Low",
        "requires_urgent_action": False,
        "clinical_complete": False,
        "report": "",
        "report_complete": False,
        "error": ""
    }
    
    # Run workflow
    final_state = app.invoke(initial_state)
    
    return final_state
