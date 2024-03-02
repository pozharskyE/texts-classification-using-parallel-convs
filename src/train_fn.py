import torch


def train(model, train_dl, loss_fn, optimizer, epochs=10, dev_dl=None, device: str = 'cuda', tr_loss_hist=[], dev_loss_hist=[], live_plot=None):
    num_train_batches = len(train_dl)

    if dev_dl:
        num_dev_batches = len(dev_dl)

    for epoch in range(epochs):
        print(f'Epoch {epoch} started')
        model.train()

        epoch_train_loss = 0

        for batch, (X_batch, y_batch) in enumerate(train_dl):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            y_pred = model(X_batch)

            loss = loss_fn(y_pred, y_batch)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_train_loss += loss.item()

            s1 = "Training: " + \
                f"[{'=' * (batch+1)}>" + \
                f"{' ' * (num_train_batches-batch-1)}]"
            print(s1, end='\r', flush=True)

        epoch_train_loss /= num_train_batches
        tr_loss_hist.append(epoch_train_loss)

        s2 = f'\rTrain loss: {epoch_train_loss}'
        print(s2 +
              (' ' * max((len(s1) - len(s2) + 1), 1)))

        if dev_dl:
            model.eval()

            epoch_dev_loss = 0

            with torch.no_grad():
                for batch, (X_batch, y_batch) in enumerate(dev_dl):
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                    y_pred = model(X_batch)

                    loss = loss_fn(y_pred, y_batch)

                    epoch_dev_loss += loss.item()

                    s1 = "Validating (on dev set):" + \
                        f"[{'=' * (batch+1)}>" + \
                        f"{' ' * (num_dev_batches-batch-1)}]"
                    print(s1, end='\r', flush=True)

            epoch_dev_loss /= num_dev_batches
            dev_loss_hist.append(epoch_dev_loss)

            s2 = f'\rDev loss: {epoch_dev_loss}'
            print(s2 +
                  (' ' * max((len(s1) - len(s2) + 1), 1)))

        print('------------')

        if live_plot:
            live_plot.update(tr_loss_hist, dev_loss_hist)

    if tr_loss_hist and not dev_loss_hist:
        return tr_loss_hist
    elif tr_loss_hist and dev_loss_hist:
        return tr_loss_hist, dev_loss_hist
