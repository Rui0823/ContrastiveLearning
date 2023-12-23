import argparse

parser = argparse.ArgumentParser(description='配置网络参数')
parser.add_argument('--patch_size', default=8, type=int, help="将最后网络输出变成多少个patch")
parser.add_argument('--batch_size', default=8, type=int, help="每次传入网络的batch")
parser.add_argument('--backbone', default="ResNet50", type=str, help="backbone")
parser.add_argument('--root', default=r'/data',type=str, help='dataset root')
parser.add_argument('--lr', default=2e-4,type=int, help='学习率')
