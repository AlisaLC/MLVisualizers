import gradio as gr
from SVM.plots import plot_kernel, plot_SVM

demo_kernel = gr.Interface(
    fn=plot_kernel,
    inputs=[
        gr.Dropdown(['linear', 'quadratic', 'gaussian'], value='gaussian'),
        gr.Dropdown(['manhattan', 'euclidean', 'maximum'], value='euclidean'),
        "checkbox"],
    outputs=["image"],
)

demo_svm = gr.Interface(
    fn=plot_SVM,
    inputs=[
        gr.Slider(),
        gr.Dropdown(['none', 'linear', 'quadratic', 'gaussian'], value='none'),
        gr.Dropdown(['manhattan', 'euclidean', 'maximum'], value='euclidean'),
        gr.Slider(1, 500, 100, step=1),
    ],
    outputs=["image"],
)