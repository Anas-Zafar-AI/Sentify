# Sentify — AI-Powered Sentiment Analyzer

> Turn customer voices into actionable insights — instantly.

## Live Demo
[Try Sentify on HuggingFace Spaces](https://huggingface.co/spaces/Anas-Zafar-AI/Sentify)

## Overview
Sentify is an enterprise-grade sentiment analysis web application that classifies any text input as Positive, Negative, or Neutral with confidence scores — enabling businesses to understand customer feedback at scale.

## Features
- Single text sentiment analysis with confidence scoring
- Bulk CSV upload — analyze thousands of reviews at once
- Downloadable results in CSV format
- Clean, professional dark-themed UI
- Powered by RoBERTa transformer model

## Tech Stack
| Component | Technology |
|-----------|------------|
| AI Model | cardiffnlp/twitter-roberta-base-sentiment-latest |
| Frontend | Gradio |
| Backend | Python |
| Deployment | HuggingFace Spaces |

## Use Cases
- E-commerce customer review analysis
- Social media sentiment monitoring
- Product feedback classification
- Customer support ticket prioritization

## How to Run Locally
```bash
git clone https://github.com/Anas-Zafar-AI/Sentify.git
cd Sentify
pip install -r requirements.txt
python app.py
```

## Project Structure
```
Sentify/
├── app.py
├── requirements.txt
└── README.md
```

## Author
**Anas Zafar** — BS Artificial Intelligence Student

[HuggingFace](https://huggingface.co/Anas-Zafar-AI) | [GitHub](https://github.com/Anas-Zafar-AI)
