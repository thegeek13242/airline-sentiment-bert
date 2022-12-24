# Script for deploying the model on Hugging Face Spaces
import gradio as gs
import infer

api = gs.Interface(fn=infer.predict, inputs="text", outputs="text")
api.launch()