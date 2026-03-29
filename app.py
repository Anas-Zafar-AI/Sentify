from transformers import pipeline
import pandas as pd
import gradio as gr

classifier = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    return_all_scores=False
)

custom_css = """
    body {
        font-family: 'Inter', sans-serif;
        background-color: #0d1117;
    }
    .gradio-container {
        max-width: 900px;
        margin: auto;
        background-color: #161b22;
        border-radius: 16px;
        padding: 40px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.5);
    }
    h1 {
        color: #e6edf3;
        font-size: 2rem;
        font-weight: 700;
        text-align: center;
        letter-spacing: -0.5px;
    }
    p {
        color: #8b949e;
        text-align: center;
        font-size: 0.95rem;
    }
    textarea, input {
        background-color: #21262d !important;
        border: 1px solid #30363d !important;
        border-radius: 10px !important;
        color: #e6edf3 !important;
        font-size: 1rem !important;
        padding: 14px !important;
    }
    textarea:focus {
        border-color: #3fb950 !important;
        box-shadow: 0 0 0 3px rgba(63,185,80,0.15) !important;
    }
    button {
        background: linear-gradient(135deg, #238636, #2ea043) !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 12px 28px !important;
        border: none !important;
        transition: opacity 0.2s ease !important;
    }
    button:hover {
        opacity: 0.85 !important;
    }
    label {
        color: #3fb950 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    .upload-container, [data-testid="file-upload"] {
        background-color: #21262d !important;
        border: 2px dashed #3fb950 !important;
        border-radius: 10px !important;
    }
"""

def analyze_single(text):
    if not text or len(str(text).strip()) < 3:
        return "NEUTRAL — Too short to analyze"
    result = classifier(text)[0]
    label = result['label'].upper()
    confidence = result['score'] * 100
    if confidence < 85:
        label = "NEUTRAL"
    return f"{label} — {confidence:.2f}% Confidence"

def analyze_csv(file):
    if file is None:
        return None, "Please upload a CSV file first."
    try:
        df = pd.read_csv(file.name)
        df.columns = df.columns.str.strip().str.lower()

        if 'review' not in df.columns:
            return None, "Error: CSV file must contain a 'review' column."

        results = []
        for text in df['review']:
            if pd.isna(text):
                results.append({'review': text, 'sentiment': 'NEUTRAL', 'confidence': '0.00%'})
                continue
            result = classifier(str(text))[0]
            label = result['label'].upper()
            confidence = result['score'] * 100
            if confidence < 85:
                label = "NEUTRAL"
            results.append({
                'review': text,
                'sentiment': label,
                'confidence': f"{confidence:.2f}%"
            })

        output_df = pd.DataFrame(results)
        output_path = "/tmp/sentiment_results.csv"
        output_df.to_csv(output_path, index=False)
        return output_path, f"Analysis complete. {len(results)} reviews processed successfully."

    except Exception as e:
        return None, f"Error: {str(e)}"

with gr.Blocks(css=custom_css, theme=gr.themes.Base()) as app:
    gr.Markdown("# Sentify")
    gr.Markdown("### Turn customer voices into actionable insights — instantly.")

    with gr.Tab("Single Text"):
        text_input = gr.Textbox(
            label="Input Text",
            placeholder="Paste any review, feedback, or sentence...",
            lines=4
        )
        text_output = gr.Textbox(label="Analysis Result", lines=2)
        analyze_btn = gr.Button("Analyze")
        analyze_btn.click(fn=analyze_single, inputs=text_input, outputs=text_output)

    with gr.Tab("Bulk CSV Upload"):
        gr.Markdown("Upload a CSV file with a column named **review**. The AI will analyze each row.")
        csv_input = gr.File(
            label="Upload CSV File",
            file_types=[".csv"],
            type="filepath"
        )
        csv_btn = gr.Button("Analyze CSV")
        csv_status = gr.Textbox(label="Status", lines=1)
        csv_output = gr.File(label="Download Results")
        csv_btn.click(fn=analyze_csv, inputs=csv_input, outputs=[csv_output, csv_status])

app.launch()