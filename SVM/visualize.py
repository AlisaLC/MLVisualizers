import gradio as gr
from SVM.plots import plot_kernel

demo_kernel = gr.Interface(
    fn=plot_kernel,
    inputs=[
        gr.Dropdown(['linear', 'quadratic', 'gaussian'], value='gaussian'),
        gr.Dropdown(['manhattan', 'euclidean', 'maximum'], value='euclidean'),
        "checkbox"],
    outputs=["image"],
)

demo_svm = gr.Interface(
    fn=plot_kernel,
    inputs=[
        gr.Slider(1, 100, 1, 1, label="C"),
        gr.Dropdown(['none', 'linear', 'quadratic', 'gaussian'], value='none'),
        gr.Dropdown(['manhattan', 'euclidean', 'maximum'], value='euclidean'),
    ],
    outputs=["image"],
)