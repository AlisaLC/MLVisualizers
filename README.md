# Machine Learning Visualizer
## Description
Machine Learning Algorithms are implemented using PyTorch and visualized using Gradio. The goal of this project is to provide a visual understanding of how machine learning algorithms work. The project is still in progress and more algorithms will be added soon.
## Installation
```
git clone https://github.com/AlisaLC/MLVisualizers.git
pip install -r requirements.txt
```
## Usage
```
python main.py --experiment [experiment_name]
```
## Experiments
### SVM
- `SVM/Kernels`: Kernel Visualization
  - Kernels:
    - Linear
    - Quadratic
    - Gaussian
  - Norms:
    - Manhattan
    - Euclidean
    - Maximum
- `SVM/SVM`: Soft Margin SVM
    - Kernels:
        - Linear
        - Quadratic
        - Gaussian
    - Norms:
        - Manhattan
        - Euclidean
        - Maximum
    - C: 0.01 - 10