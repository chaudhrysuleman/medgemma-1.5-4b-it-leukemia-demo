# LeukemiaScope — Agentic AI for Blood Cell Screening

**LeukemiaScope** is an end-to-end, multi-agent AI system that screens blood cell microscopy images for leukemia and generates professional clinical screening reports — all powered by a fine-tuned **MedGemma 1.5-4b-it** model.

## What We Built

We fine-tuned Google's MedGemma 1.5-4b-it vision-language model using **LoRA** on a curated dataset of blood cell microscopy images to classify cells as **Normal** or **Leukemia**. The fine-tuned model achieves **78% accuracy** with **83% leukemia recall**, optimized for clinical sensitivity — because missing a leukemia case is far more costly than a false positive.

But classification alone isn't enough. We built an **agentic AI pipeline** using **LangGraph** that orchestrates three specialized agents:

1. **🔬 Image Analyzer Agent** — Runs the fine-tuned MedGemma model to classify the uploaded blood cell image and extract a confidence score.
2. **🩺 Clinical Advisor Agent** — Activated only when leukemia is detected, this agent uses **Gemini 2.5 Flash** with a comprehensive hematology knowledge base to generate concise, actionable clinical recommendations and severity assessment.
3. **📋 Report Generator Agent** — Compiles all findings into a structured HTML medical report and a downloadable PDF, including patient information, classification results, clinical advice, and appropriate disclaimers.

## Key Design Decisions

- **Conditional Routing**: The Clinical Advisor is only triggered when leukemia is detected, saving latency and API costs for normal results.
- **Graceful Fallbacks**: If the Gemini API is unavailable, the system falls back to a static clinical knowledge base — ensuring reports are always generated.
- **Privacy-First**: A consent notice on the patient info page confirms that no personal data is stored or retained.

## Tech Stack

MedGemma 1.5-4b-it (LoRA fine-tuned) · Gemini 2.5 Flash · LangGraph · LangChain · Gradio · fpdf2 · HuggingFace Transformers + PEFT

## Links

- **Live Demo**: [HuggingFace Spaces](https://huggingface.co/spaces/chaudhrysuleman/Leukemia-AI)
- **Source Code**: [GitHub](https://github.com/chaudhrysuleman/medgemma-1.5-4b-it-leukemia-demo)
- **Fine-tuned LoRA Adapter**: [HuggingFace Hub](https://huggingface.co/chaudhrysuleman/medgemma-1.5-4b-it-leukemia-lora)
