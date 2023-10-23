# EXTERNAL IMPORT(S)
import argparse, os, torch

class Options:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Lung Cancer Histology Image Classifier')
        parser.add_argument('--dataset-path', type=str, default='./dataset')
        parser.add_argument('--testset-path', type=str, default='./dataset/test')
        parser.add_argument('--pretrained-path', type=str, default='./pretrained')
        parser.add_argument('--patch-stride', type=int, default=256)
        parser.add_argument('--gpu', type=str, default='0') # 0 = 1 GPU

        self._parser = parser

    def parse(self):
        opt = self._parser.parse_args()
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids

        return opt
