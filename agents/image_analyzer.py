"""
Image Analyzer Agent
Uses MedGemma to analyze blood cell images
"""

from typing import TypedDict
from PIL import Image
from tools.medgemma_predictor import get_predictor


class ImageAnalysisState(TypedDict):
    """State after image analysis"""
    image_path: str
    classification: str
    confidence: float
    is_leukemia: bool
    raw_response: str


def analyze_image(image: Image.Image) -> ImageAnalysisState:
    """
    Analyze blood cell image using MedGemma
    
    Args:
        image: PIL Image of blood cell
        
    Returns:
        ImageAnalysisState with classification results
    """
    predictor = get_predictor()
    result = predictor.predict(image)
    
    return ImageAnalysisState(
        image_path="uploaded_image",
        classification=result["classification"],
        confidence=result["confidence"],
        is_leukemia=result["is_leukemia"],
        raw_response=result["raw_response"]
    )


def image_analyzer_node(state: dict) -> dict:
    """
    LangGraph node for image analysis
    
    Args:
        state: Graph state containing 'image' key
        
    Returns:
        Updated state with analysis results
    """
    image = state.get("image")
    
    if image is None:
        return {
            **state,
            "classification": "Error",
            "confidence": 0.0,
            "is_leukemia": False,
            "analysis_error": "No image provided"
        }
    
    result = analyze_image(image)
    
    return {
        **state,
        "classification": result["classification"],
        "confidence": result["confidence"],
        "is_leukemia": result["is_leukemia"],
        "raw_response": result["raw_response"],
        "analysis_complete": True
    }
