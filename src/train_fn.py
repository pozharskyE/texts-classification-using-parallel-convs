import torch
import numpy as np
from typing import List


def train(model, train_dl, loss_fn, optimizer, epochs=10, dev_dl=None, device: str = 'cuda', tr_loss_hist: List[float] = [], dev_loss_hist: List[float] = [], calc_accuracy: bool = True, continue_epochs: bool = True, live_plot=None):

    if continue_epochs:
        num_prev_epochs = len(tr_loss_hist)

    num_train_batches = len(train_dl)
    if dev_dl:
        num_dev_batches = len(dev_dl)

    for epoch in range(epochs):
        if continue_epochs:
            print(
                f'Epoch {epoch + 1 + num_prev_epochs} results (#{epoch + 1} in this launch)')
        else:
            print(f'Epoch {epoch + 1} results')

        model.train()

        epoch_train_loss = 0

        if calc_accuracy:
            epoch_train_acc = []

        for (X_batch, y_batch) in train_dl:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            y_pred = model(X_batch)

            loss = loss_fn(y_pred, y_batch)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            model.eval()

            for (X_batch, y_batch) in train_dl:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                epoch_train_loss += loss.item()

                if calc_accuracy:
                    labels_pred = torch.tensor(
                        [round(val.item()) for val in y_pred], device='cpu')
                    y_batch = y_batch.to('cpu').reshape(-1)
                    accuracy = (labels_pred == y_batch).sum(
                    ).item() / len(y_batch)
                    epoch_train_acc.append(accuracy)

        epoch_train_loss /= num_train_batches
        tr_loss_hist.append(epoch_train_loss)

        if calc_accuracy:
            epoch_train_acc = np.array(epoch_train_acc).mean()
            epoch_train_acc = round(epoch_train_acc * 100, 3)
            print(
                f'Train: loss = {epoch_train_loss}; accuracy = {epoch_train_acc} %')
        else:
            print(f'Train loss: {epoch_train_loss}')

        if dev_dl:
            model.eval()

            epoch_dev_loss = 0

            if calc_accuracy:
                epoch_dev_acc = []

            with torch.no_grad():
                for (X_batch, y_batch) in dev_dl:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                    y_pred = model(X_batch)

                    loss = loss_fn(y_pred, y_batch)

                    epoch_dev_loss += loss.item()

                    if calc_accuracy:
                        labels_pred = torch.tensor(
                            [round(val.item()) for val in y_pred], device='cpu')
                        y_batch = y_batch.to('cpu').reshape(-1)
                        accuracy = (labels_pred == y_batch).sum(
                        ).item() / len(y_batch)
                        epoch_dev_acc.append(accuracy)

            epoch_dev_loss /= num_dev_batches
            dev_loss_hist.append(epoch_dev_loss)

            if calc_accuracy:
                epoch_dev_acc = np.array(epoch_dev_acc).mean()
                epoch_dev_acc = round(epoch_dev_acc * 100, 3)
                print(
                    f'Dev: loss = {epoch_dev_loss}; accuracy = {epoch_dev_acc} %')
            else:
                print(f'Dev loss: {epoch_dev_loss}')

        print('------------')

        if live_plot:
            live_plot.update(tr_loss_hist, dev_loss_hist)

    if tr_loss_hist and dev_loss_hist:
        return tr_loss_hist, dev_loss_hist
    else:
        return tr_loss_hist
