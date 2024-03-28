import torch
import numpy as np
from typing import List

def test(model, dataloaders, loss_fn, model_device: str, use_device='cpu', comments: List[str]=['Train', 'Dev', 'Test']):

    with torch.no_grad():

        model.eval()

        losses = []
        accuracies = []

        for dl in dataloaders:
            current_dl_losses = []
            current_dl_accuracies = []
            for batch, (X_batch, y_batch) in enumerate(dl):
                X_batch = X_batch.to(model_device)

                y_batch = y_batch.to(use_device)
                y_pred = model(X_batch).to(use_device)

                loss = loss_fn(y_pred, y_batch)

                labels_pred = torch.tensor(
                    [round(val.item()) for val in y_pred], device=use_device)
                y_batch = y_batch.reshape(-1)

                accuracy = (labels_pred == y_batch).sum().item() / len(y_batch)

                current_dl_losses.append(loss.item())
                current_dl_accuracies.append(accuracy)

            losses.append(np.array(current_dl_losses).mean())
            accuracies.append(np.array(current_dl_accuracies).mean())

    if len(comments) != len(dataloaders):
        raise ValueError(
            'len(comments) != len(dataloaders). Please, pass a string comment for each dataloader ("comments" parameter - list of strings)')

    for loss, accuracy, comment in zip(losses, accuracies, comments):
        print(comment)
        print('Loss:', loss, 'Accuracy:', round(accuracy * 100, 3), '%')
        print('------------')
