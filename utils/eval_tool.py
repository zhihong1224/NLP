import torch
from utils.base_tool import tokens2sentence,computebleu


def evaluate(model, iterator, criterion, device):
    model.eval()

    epoch_loss = 0
    bleu_score=0.
    n=0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch[0].permute(1,0)
            src_len = torch.tensor([src.shape[0]]*src.shape[1])
            trg = batch[1].permute(1,0)

            src=src.to(device)
            trg=trg.to(device)
            # src = src.to(args.local_rank)
            # trg = trg.to(args.local_rank)

            output = model(src, src_len, trg, 0)  # turn off teacher forcing

            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            preds=output.argmax(-1,keepdim=True).permute(1,0)

            preds = tokens2sentence(preds, iterator.dataset.int2word_cn)
            sources = tokens2sentence(batch[0], iterator.dataset.int2word_en)
            targets = tokens2sentence(trg.reshape(batch[0].shape[0],-1), iterator.dataset.int2word_cn)
            # for source, pred, target in zip(sources, preds, targets):
            #     source = ' '.join(source[1:])
            #     pred = ' '.join(pred)
            #     target = ' '.join(target)
            #     result.append((source, pred, target))
            # # Bleu Score
            bleu_score += computebleu(preds, targets)
            #
            n += batch[0].shape[0]

            epoch_loss += loss.item()
    return epoch_loss / len(iterator), bleu_score/n


def evaluate_transformer(model, iterator, criterion, device):
    model.eval()

    epoch_loss = 0
    bleu_score = 0.
    n = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch[0]
            trg = batch[1]
            src = src.to(device)
            trg = trg.to(device)

            output, _ = model(src, trg[:, :-1])

            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]

            loss = criterion(output, trg)

            preds = output.argmax(-1, keepdim=True).permute(1, 0)

            preds = tokens2sentence(preds, iterator.dataset.int2word_cn)
            sources = tokens2sentence(batch[0], iterator.dataset.int2word_en)
            targets = tokens2sentence(trg.reshape(batch[0].shape[0], -1), iterator.dataset.int2word_cn)

            bleu_score += computebleu(preds, targets)

            n += batch[0].shape[0]

            epoch_loss += loss.item()

    return epoch_loss / len(iterator), bleu_score/n
