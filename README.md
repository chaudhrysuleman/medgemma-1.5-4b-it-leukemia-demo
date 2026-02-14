---
title: LeukemiaScope - Agentic AI
emoji: 🩸
colorFrom: red
colorTo: pink
sdk: gradio
sdk_version: 5.12.0
python_version: "3.12"
app_file: app.py
pinned: true
license: mit
suggested_hardware: t4-small
models:
  - google/medgemma-1.5-4b-it
  - chaudhrysuleman/medgemma-1.5-4b-it-leukemia-lora
---

# 🩸 LeukemiaScope — Agentic AI

**Multi-Agent Blood Cell Analysis System powered by MedGemma + LangGraph**

> Built for the **MedGemma Impact Challenge 2026** by Chaudhry Muhammad Suleman & Muhammad Idnan

LeukemiaScope is an AI-powered medical screening tool that uses a **multi-agent workflow** to analyze microscopy images of blood cells for leukemia detection. It combines a fine-tuned vision-language model (MedGemma) with clinical reasoning (Gemini) to produce structured medical reports with actionable recommendations.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        LeukemiaScope Agentic AI                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                     Gradio Web Interface                        │  │
│  │  Step 1: Patient Info → Step 2: Image Upload → Step 3: Report  │  │
│  └──────────────────────────┬──────────────────────────────────────┘  │
│                             │                                         │
│  ┌──────────────────────────▼──────────────────────────────────────┐  │
│  │                   LangGraph Workflow Engine                     │  │
│  │                                                                 │  │
│  │  ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐   │  │
│  │  │   🔬 Image  │───▶│  🩺 Clinical │───▶│  📋 Report       │   │  │
│  │  │   Analyzer  │    │   Advisor    │    │   Generator      │   │  │
│  │  │  (MedGemma) │    │  (Gemini)    │    │  (HTML + PDF)    │   │  │
│  │  └─────────────┘    └──────────────┘    └──────────────────┘   │  │
│  │        │                    ▲                                   │  │
│  │        │    (Normal)        │ (Leukemia only)                  │  │
│  │        └────────────────────┴─────────▶ Report Generator       │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  ┌─────────────────────┐  ┌────────────────────────────────────────┐ │
│  │   Tools / Utilities │  │         External Services              │ │
│  │  • MedGemma Predict │  │  • HuggingFace Hub (model weights)    │ │
│  │  • PDF Generator    │  │  • Google AI (Gemini 3 Flash Preview)  │ │
│  └─────────────────────┘  └────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🤖 LangGraph Agent Flow

The multi-agent workflow is orchestrated by [LangGraph](https://langchain-ai.github.io/langgraph/), providing stateful execution with conditional routing:

```
                    ┌────────────────────┐
                    │    START           │
                    │  (Patient Image)   │
                    └────────┬───────────┘
                             │
                             ▼
                    ┌────────────────────┐
                    │  🔬 Image Analyzer │
                    │                    │
                    │  MedGemma 1.5 4B   │
                    │  + LoRA Adapter    │
                    │                    │
                    │  Output:           │
                    │  • Classification  │
                    │  • Confidence      │
                    │  • is_leukemia     │
                    └────────┬───────────┘
                             │
                    ┌────────▼───────────┐
                    │  Conditional Edge  │
                    │  is_leukemia?      │
                    └──┬─────────────┬───┘
                       │             │
               Yes ◀───┘             └───▶ No
                       │                    │
                       ▼                    │
               ┌──────────────────────────┐   │
               │ 🩺 Clinical Advisor       │   │
               │                           │   │
               │  Gemini 3 Flash Preview   │   │
               │  + Knowledge Base         │   │
               │                           │   │
               │  Output:                  │   │
               │  • Recommendations        │   │
               │  • Next Steps             │   │
               │  • Severity Level         │   │
               └────────────┬──────────────┘   │
                       │                    │
                       ▼                    ▼
                    ┌────────────────────────┐
                    │  📋 Report Generator   │
                    │                        │
                    │  Structured HTML Report │
                    │  + PDF Export           │
                    │                        │
                    │  Output:               │
                    │  • Medical Report      │
                    │  • Downloadable PDF    │
                    │  • Workflow Trace       │
                    └────────┬───────────────┘
                             │
                             ▼
                    ┌────────────────────┐
                    │       END          │
                    │  (Results to UI)   │
                    └────────────────────┘
```

**Key Design Decisions:**
- **Conditional Routing**: Clinical Advisor is only invoked when leukemia is detected, saving API calls and latency for normal results.
- **Shared State**: All agents read/write to a common `WorkflowState` TypedDict, ensuring data flows seamlessly.
- **Graceful Fallbacks**: If the Gemini API key is missing, the Clinical Advisor falls back to a static knowledge base response.

---

## 📂 Project Structure

```
Leukemia-AI/
├── app.py                      # Main Gradio UI + orchestration
├── requirements.txt            # Python dependencies
├── .env-example                # Example environment variables
├── README.md                   # Documentation
│
├── agents/                     # Agent definitions
│   ├── __init__.py
│   ├── image_analyzer.py       # MedGemma-based blood cell classifier
│   ├── clinical_advisor.py     # Gemini-powered clinical recommendations
│   └── report_generator.py     # HTML medical report builder
│
├── graph/                      # LangGraph workflow
│   ├── __init__.py
│   └── workflow.py             # StateGraph definition + conditional routing
│
├── tools/                      # Reusable tools
│   ├── __init__.py
│   ├── medgemma_predictor.py   # MedGemma model loading + inference
│   └── pdf_generator.py        # PDF report generation (fpdf2)
│
└── examples/                   # Sample images for testing
    ├── normal_cell.png
    └── leukemia_cell.png
```

---

## 🔬 Agent Details

### 1. Image Analyzer Agent
| Property | Value |
|----------|-------|
| **Model** | `google/medgemma-1.5-4b-it` (base) |
| **Adapter** | `chaudhrysuleman/medgemma-1.5-4b-it-leukemia-lora` |
| **Task** | Binary classification (Normal / Leukemia) |
| **Accuracy** | 77.66% |
| **Precision** | 84.16% |
| **Recall (Leukemia)** | 81.05% (optimized for sensitivity) |
| **F1-Score** | 82.57% |
| **Input** | Blood cell microscopy image (RGB) |
| **Output** | Classification, confidence score, raw response |

### 2. Clinical Advisor Agent
| Property | Value |
|----------|-------|
| **Model** | Gemini 3 Flash Preview |
| **Trigger** | Only when leukemia is detected |
| **Knowledge Base** | ALL clinical guidelines (diagnosis, risk stratification, treatment) |
| **Output** | Recommendations, next steps, severity level |
| **Fallback** | Static knowledge base response (if no API key) |

### 3. Report Generator Agent
| Property | Value |
|----------|-------|
| **Format** | Styled HTML report + downloadable PDF |
| **Content** | Patient info, classification, clinical advice, disclaimer |
| **PDF Engine** | fpdf2 |

---

## 🚀 Setup & Usage

### Prerequisites
- Python 3.9+
- [HuggingFace Token](https://huggingface.co/settings/tokens) (with access to MedGemma)
- [Google AI API Key](https://aistudio.google.com/apikey) (for Clinical Advisor)

### Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/chaudhrysuleman/medgemma-1.5-4b-it-leukemia-demo.git
   cd medgemma-1.5-4b-it-leukemia-demo
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   
   Copy the example file and add your API keys:
   ```bash
   cp .env-example .env
   ```
   Then edit `.env` and set your keys:
   ```
   HF_TOKEN=your_huggingface_token
   GOOGLE_API_KEY=your_google_api_key
   ```

4. **Run the app**
   ```bash
   python app.py
   ```

> **Note:** For HuggingFace Spaces deployment, set `HF_TOKEN` and `GOOGLE_API_KEY` in Space Secrets instead of using a `.env` file.

---

## 🖥️ User Flow

1. **Step 1 — Patient Info**: Enter patient name, date of birth, and gender
2. **Step 2 — Image Upload**: Upload a blood cell microscopy image
3. **Step 3 — Results**: View the AI analysis report with:
   - Classification result (Normal / Leukemia) with confidence
   - Clinical recommendations (if leukemia detected)
   - Downloadable PDF report
   - Workflow execution trace

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Multi-Agent Framework** | LangGraph (StateGraph) |
| **Vision Model** | MedGemma 1.5 4B-IT + LoRA |
| **Clinical LLM** | Gemini 3 Flash Preview |
| **LLM Integration** | LangChain + LangChain-Google-GenAI |
| **Web Interface** | Gradio 5.x |
| **PDF Generation** | fpdf2 |
| **Model Serving** | HuggingFace Transformers + PEFT |

---

## ⚠️ Disclaimer

> This tool is for **research and educational purposes only**. It is **NOT** a medical diagnosis tool. Results must be confirmed by qualified healthcare professionals. Do not make treatment decisions based solely on this tool's output.

---

## 👥 Authors

- **Chaudhry Muhammad Suleman**
- **Muhammad Idnan**

Built for the **MedGemma Impact Challenge 2026** 🏆

