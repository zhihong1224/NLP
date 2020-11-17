import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from pylab import *
import matplotlib as mlp
import matplotlib.pyplot as plt

from get_datasets.cmn_eng_datasets import EN2CNDataset
from utils.config import args
from utils.base_tool import computebleu, tokens2sentence, init_weights_transformer, epoch_time, init_seeds
from utils.eval_tool import evaluate_transformer
from utils.inference_tool import translate_sentence_transformer, display_attention_transformer
from models.transformer import Encoder, Decoder, Seq2Seq
from training import train_transformer


mlp.rcParams['font.sans-serif'] = ['SimHei']
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')


def main():
    import math
    import warnings
    warnings.filterwarnings('ignore')
    SEED = 1234
    init_seeds(SEED)
    # prepare dataset  and some parameters
    train_dataset = EN2CNDataset(args.data_path, args.max_output_len, 'training')
    train_loader = data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    val_dataset = EN2CNDataset(args.data_path, args.max_output_len, 'validation')
    val_loader = data.DataLoader(val_dataset, shuffle=True, batch_size=1)
    INPUT_DIM = train_dataset.en_vocab_size
    OUTPUT_DIM = train_dataset.cn_vocab_size
    SRC_PAD_IDX=train_dataset.word2int_en['<PAD>']
    TRG_PAD_IDX=train_dataset.word2int_cn['<PAD>']


    # INPUT_DIM=3922
    # OUTPUT_DIM=3805
    # SRC_PAD_IDX = 0
    # TRG_PAD_IDX = 0

    enc = Encoder(INPUT_DIM,256,3,8,512,0.1,device)
    dec = Decoder(OUTPUT_DIM, 256,3,8,512,0.1,device)
    model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
    if args.train:     # train the model
        model.apply(init_weights_transformer)
        optimizer = optim.Adam(model.parameters(),lr=5e-4)
        criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)
        N_EPOCHS = 20
        CLIP = 1
        train_losses, valid_losses, valid_bleus = train_transformer(model, N_EPOCHS, train_loader, val_loader, criterion, optimizer,
                                                        CLIP, device)

        model.load_state_dict(torch.load(f'{args.load_model_path}'))
        test_dataset = EN2CNDataset(args.data_path, args.max_output_len, 'testing')
        test_loader = data.DataLoader(test_dataset, shuffle=False, batch_size=1)
        test_loss, test_bleu = evaluate_transformer(model, test_loader, criterion, device)
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
        trans_start=time.time()
        sentence = "I'll buy an apple for you ."
        pre, att, show_used_sen = translate_sentence_transformer(sentence, model, device, max_len=args.max_output_len)
        trans_end=time.time()
        print('\nsrc:', sentence, ' translation:', ''.join(pre[:-1]))
        print('translation time:',trans_end-trans_start)
        display_attention_transformer(show_used_sen, pre, att)


if __name__ == '__main__':
    main()
