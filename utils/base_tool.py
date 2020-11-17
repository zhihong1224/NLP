import os
import random
import torch
import torch.nn as nn
import numpy as np
# from models.rnn import Encoder, Decoder, Seq2Seq
import torch.distributed as dist
from utils.config import args

from nltk.translate.bleu_score import sentence_bleu


def init_seeds(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def init_weights_transformer(m):
    if hasattr(m,'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def save_model(model, optimizer, store_model_path, step):
    if args.local_rank == 0:
        torch.save(model.module.state_dict(), f'{store_model_path}/model_{step}.ckpt')
        return


def load_model(model, load_model_path):
    print(f'Load model from {load_model_path}')
    filename=os.listdir(load_model_path)
    load_model_path=os.path.join(load_model_path,filename[-1])
    model.load_state_dict(torch.load(f'{load_model_path}'))
    return model


def tokens2sentence(outputs, int2word):
    sentences = []
    for tokens in outputs:
        sentence = []
        for token in tokens:
            word = int2word[str(int(token))]
            if word == '<EOS>':
                break
            sentence.append(word)
        sentences.append(sentence)

    return sentences


def computebleu(sentences, targets):
    score = 0
    assert (len(sentences) == len(targets))

    def cut_token(sentence):
        tmp = []
        for token in sentence:
            if token == '<UNK>' or token.isdigit() or len(bytes(token[0], encoding='utf-8')) == 1:
                tmp.append(token)
            else:
                tmp += [word for word in token]
        return tmp

    for sentence, target in zip(sentences, targets):
        sentence = cut_token(sentence)
        target = cut_token(target)
        # score += sentence_bleu([target], sentence, weights=(1, 0, 0, 0))
        score += sentence_bleu([target], sentence)

    return score


def reduce_tensor(tensor: torch.Tensor):
    rt = torch.clone(tensor)
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def schedule_sampling(step, summary_steps, c, k):
    if c == 0:
        e = 1
    if c == 1:
        e = k / (k + np.exp(step / k))
    elif c == 2:
        e = -1 / summary_steps * step + 1
    elif c == 3:
        e = np.power(0.9, step)
    return e
