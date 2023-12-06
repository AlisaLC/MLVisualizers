import argparse
from SVM.visualize_kernel import visualize_kernels

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('experiment', type=str, help='Experiment name')
    args = parser.parse_args()

    if args.experiment == 'SVM/Kernels':
        visualize_kernels()