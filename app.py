# Script for deploying the model on Hugging Face Spaces

import gradio as gr
import infer

api = gr.Interface(fn=infer.predict, inputs=gr.Textbox(
    lines=2, placeholder="Enter input sentence here", label="Sentence"), outputs=gr.Textbox(label="Sentiment"))
api.launch()
