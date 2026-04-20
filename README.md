# IndustryGPT — Financial Literacy Chatbot

An AI-powered chatbot fine-tuned on domain-specific financial education data to provide personalized guidance on budgeting, saving, credit management, investment strategies, and retirement planning.

---

## Overview

Access to reliable, personalized financial advice remains limited for most individuals. IndustryGPT addresses this by fine-tuning **Llama-2-7B** on a curated financial literacy dataset using **QLoRA** — a parameter-efficient fine-tuning technique that makes large language model training feasible on consumer-grade hardware through 4-bit quantization.

The project covers the full pipeline: data collection and OCR-based extraction, preprocessing, fine-tuning, and serving the model through a Flask web interface.

---

## Tech Stack

| Component | Technology |
|---|---|
| Base Model | Llama-2-7B (Meta) |
| Fine-tuning Method | QLoRA (Quantized Low-Rank Adaptation) |
| Quantization | 4-bit NF4 via `bitsandbytes` |
| Training Framework | Hugging Face `transformers`, `peft`, `trl` |
| OCR | `pytesseract`, `pdf2image` |
| Web Interface | Flask, HTML/CSS |
| Tunneling | Ngrok |

---

## Pipeline

```
PDF/Image Sources
      │
      ▼
OCR Extraction (pytesseract)
      │
      ▼
Dataset Preparation (Hugging Face + GitHub sources)
      │
      ▼
Prompt-Response Formatting
      │
      ▼
QLoRA Fine-tuning (Llama-2-7B)
      │
      ▼
Flask Web App → Ngrok → Public URL
```

---

## Model Configuration

| Parameter | Value |
|---|---|
| LoRA Rank | 64 |
| LoRA Alpha (Scaling) | 16 |
| LoRA Dropout | 0.1 |
| Quantization | 4-bit NF4 |
| Training Epochs | 1 |
| Batch Size | 4 |
| Learning Rate | 2e-4 |
| Optimizer | AdamW |

---

## Key Design Decisions

**Why QLoRA over full fine-tuning?**
Full fine-tuning of a 7B parameter model requires 80GB+ of GPU VRAM. QLoRA reduces this to under 12GB by freezing the base model weights and training small low-rank adapter matrices in 4-bit precision — making it practical without cloud-scale compute.

**Why Llama-2-7B over GPT-2?**
GPT-2 (117M–1.5B parameters) lacks the reasoning capacity needed for coherent multi-turn financial Q&A. Llama-2-7B was chosen for its stronger instruction-following capability and open-weights availability for fine-tuning.

**OCR Pipeline**
Financial literacy resources often exist as scanned PDFs. A custom OCR pipeline using `pytesseract` and `pdf2image` was built to extract and clean text from image-based documents before training.

---

## Running the Chatbot

### Prerequisites
- Python 3.9+
- GPU with 12GB+ VRAM (recommended)
- Hugging Face account with Llama-2 access approved

### Installation

```bash
git clone https://github.com/Satyam-G-Kulkarni/IndustryGPT---Financial-Literacy-and-Personal-Finance-Management.git
cd IndustryGPT---Financial-Literacy-and-Personal-Finance-Management
pip install -r requirements.txt
```

### Run the Flask App

```bash
python financegpt_app.py
```

The app starts locally and Ngrok generates a public URL for external access.

---

## Topics Covered by the Chatbot

- Budgeting and expense tracking
- Saving strategies and emergency funds
- Credit scores and debt management
- Investment fundamentals (stocks, mutual funds, SIPs)
- Retirement planning and tax-saving instruments

---

## Limitations

- Responses are based on the fine-tuning dataset and may not reflect the most current financial regulations or products
- The model was trained for 1 epoch due to compute constraints — additional training epochs would improve response relevance
- Not a substitute for professional financial advice

---

## Future Work

- Increase training epochs and evaluate with ROUGE/BLEU metrics
- Integrate RAG (Retrieval-Augmented Generation) for real-time data grounding
- Add conversation memory for multi-turn context retention
- Deploy on cloud (AWS/GCP) with a persistent endpoint

---

## Author

**Satyam Kulkarni**
ML Engineer | MSc AI & ML
[LinkedIn](https://www.linkedin.com/in/satyam-kulkarni-92004215b/) • [GitHub](https://github.com/Satyam-G-Kulkarni)
