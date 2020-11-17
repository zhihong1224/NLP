import argparse


parser=argparse.ArgumentParser(description='Translation_Training')
# parser.add_argument('--local_rank',default=-1,type=int)
parser.add_argument('--batch_size',default=60,type=int)
parser.add_argument('--emb_dim',default=256,type=int)
parser.add_argument('--hid_dim',default=512,type=int)
parser.add_argument('--dropout',default=0.5,type=float)
parser.add_argument('--max_output_len',default=50,type=int)
parser.add_argument('--data_path',default="./cmn-eng")

# args that need adjust per train----------------
parser.add_argument('--train',default=False)
parser.add_argument('--attention',default=True)
parser.add_argument('--c',default=3)
parser.add_argument('--store_model_path',default='pretrained/rnn_att_sampling.pt')
parser.add_argument('--load_model_path',default='pretrained/rnn_multi_heads_sampling.pt')

args=parser.parse_args()
