import gradio as gr
from plots import plot_kernel

demo = gr.Interface(
    fn=plot_kernel,
    inputs=[
        gr.Dropdown(['linear', 'quadratic', 'gaussian'], value='gaussian'),
        gr.Dropdown(['manhattan', 'euclidean', 'maximum'], value='euclidean'),
        "checkbox"],
    outputs=["image"],
)

demo.launch(share=True)