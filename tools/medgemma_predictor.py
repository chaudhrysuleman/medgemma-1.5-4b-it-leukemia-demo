"""
MedGemma Predictor Tool
Uses the fine-tuned LoRA model for leukemia classification
"""

import os
import torch
from PIL import Image
from typing import Optional
from langchain_core.tools import BaseTool
from pydantic import Field


class MedGemmaPredictor:
    """Wrapper for the fine-tuned MedGemma model"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        self.loaded = False
        
        # Model IDs
        self.base_model_id = "google/medgemma-1.5-4b-it"
        self.lora_adapter_id = "chaudhrysuleman/medgemma-1.5-4b-it-leukemia-lora"
    
    def load(self, hf_token: Optional[str] = None):
        """Load the model and processor"""
        if self.loaded:
            return True
            
        from transformers import AutoProcessor, AutoModelForImageTextToText
        from peft import PeftModel
        from huggingface_hub import login
        
        token = hf_token or os.getenv("HF_TOKEN", "")
        token = token.strip()  # Remove trailing newline from Spaces secrets
        if token:
            login(token=token)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📍 Loading MedGemma on {self.device}")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.base_model_id, 
            token=token
        )
        
        # Load base model
        base_model = AutoModelForImageTextToText.from_pretrained(
            self.base_model_id,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            token=token
        )
        
        # Load LoRA adapters
        self.model = PeftModel.from_pretrained(
            base_model, 
            self.lora_adapter_id, 
            token=token
        )
        self.model.eval()
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        self.loaded = True
        print("✅ MedGemma loaded successfully!")
        return True
    
    def predict(self, image: Image.Image) -> dict:
        """
        Predict leukemia from blood cell image
        
        Args:
            image: PIL Image of blood cell
            
        Returns:
            dict with classification, confidence, raw_response
        """
        if not self.loaded:
            self.load()
        
        # Ensure RGB
        image = image.convert("RGB")
        
        # Classification prompt (same as training)
        prompt = (
            "Analyze this blood cell microscopy image and classify it.\n"
            "Is the cell NORMAL or LEUKEMIA (blast)?\n"
            "Answer with exactly one of: Normal, Leukemia.\n"
        )
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Process
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt"
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
            )
        
        # Decode
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        last_line = response.strip().split('\n')[-1].strip().lower()
        
        # Parse result
        if "normal" in last_line and "leukemia" not in last_line:
            classification = "Normal"
            confidence = 0.70  # Approximate based on model performance
        elif "leukemia" in last_line:
            classification = "Leukemia"
            confidence = 0.83  # Based on model recall
        else:
            classification = "Uncertain"
            confidence = 0.50
        
        return {
            "classification": classification,
            "confidence": confidence,
            "raw_response": last_line,
            "is_leukemia": classification == "Leukemia"
        }


# Global instance
_predictor: Optional[MedGemmaPredictor] = None


def get_predictor() -> MedGemmaPredictor:
    """Get or create the global predictor instance"""
    global _predictor
    if _predictor is None:
        _predictor = MedGemmaPredictor()
    return _predictor


def predict_image(image: Image.Image) -> dict:
    """Convenience function to predict from image"""
    predictor = get_predictor()
    return predictor.predict(image)
