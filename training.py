import time
import math
import torch
from utils.eval_tool import evaluate_transformer,evaluate
from utils.base_tool import epoch_time, schedule_sampling
from utils.config import args


def train_epoch(step, N_EPOCHS, model, iterator, optimizer, criterion, clip, device):
    model.train()

    epoch_loss = 0
    p = schedule_sampling(step, N_EPOCHS, c=args.c, k=1)
    print('Epoch: 0{} | p: {}'.format(step,p))
    for i, batch in enumerate(iterator):
        src=batch[0].permute(1,0)
        src_len = torch.tensor([src.shape[0]]*src.shape[1])
        trg = batch[1].permute(1,0)

        optimizer.zero_grad()
        src=src.to(device)
        trg=trg.to(device)

        # use schedule sampling ------------------------------------------------------------
        # output = model(src, src_len, trg)

        output = model(src, src_len, trg, teacher_forcing_ratio=p)

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def train_epoch_transformer(model,iterator,optimizer,criterion,clip,device):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch[0]
        trg = batch[1]
        src = src.to(device)
        trg = trg.to(device)

        optimizer.zero_grad()

        output, _ = model(src, trg[:, :-1])

        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def train(model,N_EPOCHS,train_loader,val_loader,criterion,optimizer,CLIP,device):
    best_valid_loss = float('inf')

    train_losses, valid_losses, valid_bleus = [], [], []

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        # use schedule sampling ---------------------------------------------------------------
        # train_loss = train_epoch(model,train_loader,optimizer,criterion,CLIP,device)
        train_loss = train_epoch(epoch, N_EPOCHS, model, train_loader, optimizer, criterion, CLIP, device)
        valid_loss, valid_bleu = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid_bleus.append(valid_bleu)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            # torch.save(model.state_dict(), 'tut5-model.pt')
            torch.save(model.state_dict(), f'{args.store_model_path}')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f} | Val. BLEU: {valid_bleu:7.3f}')
    return train_losses, valid_losses, valid_bleus


def train_transformer(model, N_EPOCHS, train_loader, val_loader, criterion, optimizer, CLIP, device):
    best_valid_loss = float('inf')

    train_losses, valid_losses, valid_bleus = [], [], []

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss = train_epoch_transformer(model, train_loader, optimizer, criterion, CLIP, device)
        valid_loss, valid_bleu = evaluate_transformer(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid_bleus.append(valid_bleu)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            # torch.save(model.state_dict(), 'tut5-model.pt')
            torch.save(model.state_dict(), f'{args.store_model_path}')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f} | Val. BLEU: {valid_bleu:7.3f}')
    return train_losses, valid_losses, valid_bleus
