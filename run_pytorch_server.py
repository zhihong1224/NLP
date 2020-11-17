import os
import re
import json
import random
import numpy as np
import flask
import torch
from nltk.tokenize import word_tokenize
from models.transformer import Encoder,Decoder,Seq2Seq
import models.rnn as r0
import models.rnn_self_att as r


def init_seeds(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


if torch.cuda.is_available():
    use_gpu=True
    device=torch.device('cuda')
else:
    use_gpu=False
    device=torch.device('cpu')

SEED = 1234
init_seeds(SEED)

INPUT_DIM = 3922
OUTPUT_DIM = 3805
SRC_PAD_IDX = 0
TRG_PAD_IDX = 0

t_enc = Encoder(INPUT_DIM, 256, 3, 8, 512, 0.1, device)
t_dec = Decoder(OUTPUT_DIM, 256, 3, 8, 512, 0.1, device)
t_model = Seq2Seq(t_enc, t_dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

attn=r.Attention(512, 512, device)
r_enc=r.Encoder(INPUT_DIM, 256, 512, 512, 0.5)
r_dec=r.Decoder(OUTPUT_DIM, 256, 512, 512, 0.5, attn)
r_model=r.Seq2Seq(r_enc, r_dec, SRC_PAD_IDX, device).to(device)

attn0=r0.Attention(512, 512)
r0_enc=r0.Encoder(INPUT_DIM, 256, 512, 512, 0.5)
r0_dec=r0.Decoder(OUTPUT_DIM, 256, 512, 512, 0.5, attn0)
r0_model=r0.Seq2Seq(r0_enc, r0_dec, SRC_PAD_IDX, device).to(device)


def load_model(model,load_model_path):
    # global model
    if use_gpu:
        model.load_state_dict(torch.load(f'{load_model_path}'))
    else:
        pretrain = torch.load(load_model_path, map_location=lambda storage, loc: storage)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in pretrain.items():
            if k == 'state_dict':
                state_dict = OrderedDict()
                for keys in v:
                    name = keys[7:]  # remove `module.`
                    state_dict[name] = v[keys]
                new_state_dict[k] = state_dict
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict['state_dict'])


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


word2int_en, int2word_en = get_dictionary('./cmn-eng', 'en')
word2int_cn, int2word_cn = get_dictionary('./cmn-eng', 'cn')


def prepare_data(mode,input):
    global word2int_en
    sentence = ''
    for word in word_tokenize(input):
        sentence += word.lower() + ' '

    # run .sh file to get tokenize
    # sentence="have you eaten your la@@ un@@ ch y@@ e@@ t@@ ?"

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
    transform = LabelTransform(50, word2int_en['<PAD>'])
    en = transform(en)
    src_tensor = torch.LongTensor([en]).to(device)  # (1, max_output_len)
    # print('en tensor:',src_tensor)
    # print(src_tensor.shape)
    if mode=='transformer':
        src_mask = t_model.make_src_mask(src_tensor)
    else:
        src_mask=None
    src_len = torch.LongTensor([en.shape[0]]).to(device)
    print(en.shape[0],src_tensor.shape[0])
    return src_tensor,src_mask,src_len


def r_inference(model,src_tensor,src_len):
    model.eval()
    # src_tensor, _, src_len = prepare_data(eng_sentence)
    src_tensor=src_tensor.permute(1,0)
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_len)

    mask = model.create_mask(src_tensor)

    trg_indexes = [word2int_cn['<BOS>']]

    attentions = torch.zeros(50, 1, 50).to(device)

    for i in range(50):

        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)

        with torch.no_grad():
            output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask)

        attentions[i] = attention

        pred_token = output.argmax(1).item()

        trg_indexes.append(pred_token)

        if pred_token == word2int_cn['<EOS>']:
            break
    trg_tokens = [int2word_cn[str(i)] for i in trg_indexes]
    return trg_tokens


def t_inference(model,src_tensor,src_mask):
    model.eval()
    # src_tensor, src_mask, _ = prepare_data(eng_sentence)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [word2int_cn['<BOS>']]

    for i in range(50):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(2)[:, -1].item()

        trg_indexes.append(pred_token)

        if pred_token == word2int_cn['<EOS>']:
            break

    trg_tokens = [int2word_cn[str(i)] for i in trg_indexes]
    return trg_tokens


app=flask.Flask(__name__,template_folder='templates')


@app.route('/predict',methods=['POST','GET'])
def predict():
    # print(flask.request.method)
    if flask.request.method=='POST':
        print('post ok')
        eng_sentence = flask.request.form.get("data_input",type=str)
        mode=flask.request.form.get("model",type=str)
        # mode='transformer'
    elif flask.request.method=='GET':
        print('get ok')
        eng_sentence = flask.request.args.get("data_input",type=str)
        mode=flask.request.args.get("model",type=str)
        # mode='transformer'
        return flask.render_template('predict.html')
    print(eng_sentence)
    print(mode)

    src_tensor, src_mask, src_len = prepare_data(mode,eng_sentence)

    if mode=='transformer':
        trg_tokens=t_inference(t_model,src_tensor,src_mask)
    elif mode=='rnn_self_attn':
        trg_tokens=r_inference(r_model,src_tensor,src_len)
    elif mode=='rnn_attn':
        trg_tokens=r_inference(r0_model,src_tensor,src_len)
    r={'input_sentence':eng_sentence,'translation_sentence':''.join(trg_tokens[1:-1])}
    return flask.render_template('predict.html',**r)


if __name__ == '__main__':
    print("Loading Pytorch model and Flask starting server...")
    print("Please wait until server has fully started")
    load_t_model_path = 'pretrained/trans.pt'
    load_model(t_model,load_t_model_path)
    # load_r_model_path='pretrained/rnn_att.pt'
    load_r_model_path='pretrained/rnn_multi_heads_sampling.pt'
    load_model(r_model,load_r_model_path)
    load_r0_model_path='pretrained/rnn_att.pt'
    load_model(r0_model,load_r0_model_path)

    app.run(debug=True)
