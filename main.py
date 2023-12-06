import argparse
from SVM.visualize import demo_kernel as SVM_Kernels_demo
from SVM.visualize import demo_svm as SVM_SVM_demo

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--experiment', type=str, help='Experiment name')
    args = parser.parse_args()

    if args.experiment == 'SVM/Kernels':
        SVM_Kernels_demo.launch(share=True)
    elif args.experiment == 'SVM/SVM':
        SVM_SVM_demo.launch(share=True)