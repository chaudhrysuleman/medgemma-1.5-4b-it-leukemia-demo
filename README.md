---
title: LeukemiaScope - Agentic AI
emoji: 🩸
colorFrom: red
colorTo: pink
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: true
license: mit
suggested_hardware: t4-small
models:
  - google/medgemma-1.5-4b-it
  - chaudhrysuleman/medgemma-1.5-4b-it-leukemia-lora
---

# 🩸 LeukemiaScope — Agentic AI

**Multi-Agent Blood Cell Analysis powered by MedGemma + LangGraph**

Built for the **MedGemma Impact Challenge 2026** by Chaudhry Muhammad Suleman & Muhammad Idnan

## Agents
- 🔬 **Image Analyzer** — Fine-tuned MedGemma LoRA
- 🩺 **Clinical Advisor** — Gemini 3 Flash Preview
- 📋 **Report Generator** — HTML + PDF reports

## Setup
Set `HF_TOKEN` and `GOOGLE_API_KEY` in Space Secrets.
