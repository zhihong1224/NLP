import math
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from pylab import *
import matplotlib as mlp
import matplotlib.pyplot as plt

from get_datasets.cmn_eng_datasets import EN2CNDataset
from utils.config import args
from utils.base_tool import computebleu, tokens2sentence, init_weights, epoch_time, init_seeds
from utils.eval_tool import evaluate
from utils.inference_tool import translate_sentence, display_attention
from models.rnn_self_att import Encoder, Decoder, Seq2Seq, Attention
from training import train


mlp.rcParams['font.sans-serif'] = ['SimHei']
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def main():
    warnings.filterwarnings('ignore')
    SEED = 1234
    init_seeds(SEED)
    # prepare dataset
    train_dataset = EN2CNDataset(args.data_path, args.max_output_len, 'training')
    train_loader = data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    val_dataset = EN2CNDataset(args.data_path, args.max_output_len, 'validation')
    val_loader = data.DataLoader(val_dataset, shuffle=True, batch_size=1)
    # prepare model
    INPUT_DIM = train_dataset.en_vocab_size
    OUTPUT_DIM = train_dataset.cn_vocab_size
    SRC_PAD_IDX = train_dataset.word2int_en['<PAD>']

    if args.attention:
        attn = Attention(args.hid_dim, args.hid_dim,device)
    else:
        attn = None
    enc = Encoder(INPUT_DIM, args.emb_dim, args.hid_dim, args.hid_dim, args.dropout)
    dec = Decoder(OUTPUT_DIM, args.emb_dim, args.hid_dim, args.hid_dim, args.dropout, attn)
    model = Seq2Seq(enc, dec, SRC_PAD_IDX, device).to(device)

    if args.train:     # train the model
        model.apply(init_weights)
        optimizer = optim.Adam(model.parameters())
        TRG_PAD_IDX = train_dataset.word2int_cn['<PAD>']
        criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)
        N_EPOCHS = 20
        CLIP = 1
        train_losses, valid_losses, valid_bleus = train(model, N_EPOCHS, train_loader, val_loader, criterion, optimizer,
                                                        CLIP, device)
        # test_dataset = EN2CNDataset(args.data_path, args.max_output_len, 'testing')
        # test_loader = data.DataLoader(test_dataset, shuffle=False, batch_size=1)
        # test_loss, test_bleu = evaluate(model, test_loader, criterion, device)
        # print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} | Test BLEU: {test_bleu:7.3f}')

        # model.load_state_dict(torch.load(f'{args.load_model_path}'))
        test_dataset = EN2CNDataset(args.data_path, args.max_output_len, 'testing')
        test_loader = data.DataLoader(test_dataset, shuffle=False, batch_size=1)
        test_loss, test_bleu = evaluate(model, test_loader, criterion, device)
        print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} | Test BLEU: {test_bleu:7.3f}')

        plt.plot(range(len(train_losses)), train_losses)
        plt.plot(range(len(valid_losses)), valid_losses, c='r')
        plt.title('loss')
        plt.show()
        plt.plot(range(len(valid_bleus)), valid_bleus)
        plt.title('bleu score')
        plt.show()

    else:     # use the pretrained model to inference
        # model.load_state_dict(torch.load('tut4-model.pt'))
        model.load_state_dict(torch.load(f'{args.load_model_path}'))
        sentence = "Tom drank my apple juice."
        pre, att, show_used_sen = translate_sentence(sentence, model, device, max_len=args.max_output_len)
        print('\nsrc:', sentence, ' translation:', ''.join(pre[:-1]))
        display_attention(show_used_sen, pre, att)


if __name__ == '__main__':
    main()

