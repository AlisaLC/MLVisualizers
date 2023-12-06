import argparse
from SVM.visualize_kernel import demo as SVM_Kernels_demo

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--experiment', type=str, help='Experiment name')
    args = parser.parse_args()

    if args.experiment == 'SVM/Kernels':
        SVM_Kernels_demo.launch(share=True)