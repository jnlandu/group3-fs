from args import *
import copy
import numpy as np
from tqdm import tqdm
import torch
from torch.autograd import Variable




def train(model,criterion,optimizer,scheduler,   num_epochs, train_dl, valid_dl, device):

    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    min_valid_loss = np.inf

    for epoch in tqdm(range(num_epochs)):

        model.train()



        for x_batch, y_batch in train_dl:

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            x, y_a, y_b, lam = mixup_data(x_batch, y_batch, alpha=0.2)  # adjust alpha as needed
            x, y_a, y_b = map(Variable, (x, y_a, y_b))

            optimizer.zero_grad()

            loss = mixup_criterion(criterion, pred, y_a, y_b, lam)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

            loss_hist_train[epoch] += loss.item() * y_batch.size(0)
            is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
            accuracy_hist_train[epoch] += is_correct.sum().item()


        loss_hist_train[epoch] /= len(train_dl.dataset)
        accuracy_hist_train[epoch] /= len(train_dl.dataset)

        scheduler.step(loss)

        model.eval()

        with torch.no_grad():

            for x_batch, y_batch in valid_dl:

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                pred = model(x_batch)
                loss = criterion(pred, y_batch)
                loss_hist_valid[epoch] += loss.item() * y_batch.size(0)
                is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
                accuracy_hist_valid[epoch] += is_correct.sum().item()

        loss_hist_valid[epoch] /= len(valid_dl.dataset)
        accuracy_hist_valid[epoch] /= len(valid_dl.dataset)

        if accuracy_hist_valid[epoch] > best_acc:
            best_acc = accuracy_hist_valid[epoch]
            best_model_wts = copy.deepcopy(model.state_dict())

        print(f'Epoch {epoch+1}:   Train accuracy: {accuracy_hist_train[epoch]:.4f}    Validation accuracy: {accuracy_hist_valid[epoch]:.4f} ')


        if loss_hist_valid[epoch] < min_valid_loss:
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            break


    model.load_state_dict(best_model_wts)

    history = {}
    history['loss_hist_train'] = loss_hist_train
    history['loss_hist_valid'] = loss_hist_valid
    history['accuracy_hist_train'] = accuracy_hist_train
    history['accuracy_hist_valid'] = accuracy_hist_valid

    return model, history

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)