
from transformers import pipeline
import gradio as gr

generator = pipeline(
    "text-generation",
    model="distilgpt2"
)

def generate_text(prompt):
    result = generator(
        prompt,
        max_length=100,
        do_sample=True,
        temperature=0.7
    )

    return result[0]["generated_text"]

demo = gr.Interface(
    fn=generate_text,
    inputs=gr.Textbox(
        lines=2,
        placeholder="Enter your prompt here..."
    ),
    outputs="text",
    title="GPT Text Generator",
    description="Generate meaningful text using DistilGPT2."
)

demo.launch(server_name="0.0.0.0", server_port=7860)
