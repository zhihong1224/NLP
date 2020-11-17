import os
import re
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from nltk.tokenize import word_tokenize
from utils.config import args


def get_dictionary(root,language):
    # load dict
    with open(os.path.join(root, f'word2int_{language}.json'), "r") as f:
        word2int = json.load(f)
    with open(os.path.join(root, f'int2word_{language}.json'), "r") as f:
        int2word = json.load(f)
    return word2int, int2word


class LabelTransform(object):
    def __init__(self, size, pad):
        self.size = size
        self.pad = pad

    def __call__(self, label):
        label = np.pad(label, (0, (self.size - label.shape[0])), mode='constant', constant_values=self.pad)
        return label


def translate_sentence(input, model, device, max_len=args.max_output_len):
    model.eval()

    sentence = ''
    for word in word_tokenize(input):
        sentence += word.lower() + ' '

    # run .sh file to get tokenize
    # sentence="have you eaten your la@@ un@@ ch y@@ e@@ t@@ ?"

    word2int_en, int2word_en=get_dictionary('./cmn-eng','en')
    word2int_cn, int2word_cn = get_dictionary('./cmn-eng', 'cn')

    sentence = re.split('[\t\n]', sentence)
    sentence = list(filter(None, sentence))

    # special words
    BOS = word2int_en['<BOS>']
    EOS = word2int_en['<EOS>']
    UNK = word2int_en['<UNK>']

    # add <EOS> and <BOS>
    en, cn = [BOS], [BOS]
    # subword
    sentence = re.split(' ', sentence[0])
    sentence = list(filter(None, sentence))
    display_used_sen = sentence     # list of tokens, will be used in plot attention
    for word in sentence:
        en.append(word2int_en.get(word, UNK))
    en.append(EOS)

    en = np.asarray(en)

    # <PAD> to same length
    transform = LabelTransform(args.max_output_len, word2int_en['<PAD>'])
    en = transform(en)
    src_tensor = torch.LongTensor([en]).permute(1,0).to(device)   #(max_output_len,1)
    # print('en tensor:',src_tensor)
    # print(src_tensor.shape)

    src_len=torch.LongTensor([en.shape[0]]).to(device)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_len)

    mask = model.create_mask(src_tensor)

    trg_indexes = [word2int_cn['<BOS>']]

    attentions = torch.zeros(max_len, 1, en.shape[0]).to(device)

    for i in range(max_len):

        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)

        with torch.no_grad():
            output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask)

        attentions[i] = attention

        pred_token = output.argmax(1).item()

        trg_indexes.append(pred_token)

        if pred_token == word2int_cn['<EOS>']:
            break
    trg_tokens = [int2word_cn[str(i)] for i in trg_indexes]

    return trg_tokens[1:], attentions[:len(trg_tokens) - 1,:,:(len(display_used_sen)+2)], display_used_sen


def translate_sentence_transformer(input, model, device, max_len=args.max_output_len):
    model.eval()

    sentence = ''
    for word in word_tokenize(input):
        sentence += word.lower() + ' '

    # run .sh file to get tokenize
    # sentence="have you eaten your la@@ un@@ ch y@@ e@@ t@@ ?"

    word2int_en, int2word_en = get_dictionary('./cmn-eng', 'en')
    word2int_cn, int2word_cn = get_dictionary('./cmn-eng', 'cn')

    sentence = re.split('[\t\n]', sentence)
    sentence = list(filter(None, sentence))

    # special words
    BOS = word2int_en['<BOS>']
    EOS = word2int_en['<EOS>']
    UNK = word2int_en['<UNK>']

    # add <EOS> and <BOS>
    en, cn = [BOS], [BOS]
    # subword
    sentence = re.split(' ', sentence[0])
    sentence = list(filter(None, sentence))
    display_used_sen = sentence  # list of tokens, will be used in plot attention
    for word in sentence:
        en.append(word2int_en.get(word, UNK))
    en.append(EOS)

    en = np.asarray(en)

    # <PAD> to same length
    transform = LabelTransform(args.max_output_len, word2int_en['<PAD>'])
    en = transform(en)
    src_tensor = torch.LongTensor([en]).to(device)  # (1, max_output_len)
    # print('en tensor:',src_tensor)
    # print(src_tensor.shape)
    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [word2int_cn['<BOS>']]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(2)[:, -1].item()

        trg_indexes.append(pred_token)

        if pred_token == word2int_cn['<EOS>']:
            break

    trg_tokens = [int2word_cn[str(i)] for i in trg_indexes]
    print(attention.shape,len(display_used_sen),len(trg_tokens))

    return trg_tokens[1:], attention[:,:,:len(trg_tokens) - 1,:(len(display_used_sen)+2)], display_used_sen


def display_attention(sentence, translation, attention):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    attention = attention.squeeze(1).cpu().detach().numpy()
    cax = ax.matshow(attention, cmap='bone')

    ax.tick_params(labelsize=15)
    ax.set_xticklabels([''] + ['<bos>'] + [t.lower() for t in sentence] + ['<eos>'],
                       rotation=45)
    ax.set_yticklabels([''] + translation)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()


def display_attention_transformer(sentence, translation, attention, n_heads=8, n_rows=4, n_cols=2):
    assert n_rows * n_cols == n_heads

    fig = plt.figure(figsize=(15, 25))

    for i in range(n_heads):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)

        _attention = attention.squeeze(0)[i].cpu().detach().numpy()

        cax = ax.matshow(_attention, cmap='bone')

        ax.tick_params(labelsize=12)
        ax.set_xticklabels([''] + ['<sos>'] + [t.lower() for t in sentence] + ['<eos>'],
                           rotation=45)
        ax.set_yticklabels([''] + translation)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()
