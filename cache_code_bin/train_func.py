import torch

def train():
  for epoch in range(epochs):
    model.train()
    print(f'Epoch {epoch+1}')
    optimizer.zero_grad()

    for (X_batch, y_batch) in train_dl:
      batch_outputs = model(X_batch)
      loss_batch = loss_fn(batch_outputs, y_batch)
      loss_batch.backward()
      optimizer.step()

    with torch.no_grad():
      model.eval()
      # measure train and dev loss at the end of epoch
      train_outputs = model(X_train)
      train_loss = loss_fn(train_outputs, y_train)
      train_loss_histroty.append(train_loss.item())

      dev_outputs = model(X_dev)
      dev_loss = loss_fn(dev_outputs, y_dev)
      dev_loss_history.append(dev_loss.item())

      print(f'Train loss: {train_loss}; Dev loss: {dev_loss}')
      live_plot.update(train_loss_histroty, dev_loss_history)