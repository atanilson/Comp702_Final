"""
Function for training and evaluating model
"""
import torch
from torch import nn
from tqdm.auto import tqdm
from typing import Dict, List
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device=device):

  # Putting the model in training mode
  model.train()

  # Setting the loss and accurecy value to accumate in each batch
  train_loss, train_acc = 0,0

  # Looping through datalaoder data batches
  for batch, (X, y) in enumerate(dataloader):
    # Sending the data to the target device
    X, y = X.to(device), y.to(device)

    # Forward pass <- logits
    y_pred = model(X)

    # Calculating the loss
    loss = loss_fn(y_pred, y)

    # Accumulating loss
    train_loss += loss.item()

    # Zero the gradient not to accumalte
    optimizer.zero_grad()

    # Back propagation - calculate the gradient
    loss.backward()

    # update the weights
    optimizer.step()

    # Calculating the accurecy logit -> pred probability -> labels
    y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
    train_acc += (y_pred_class==y).sum().item()/len(y_pred)

  # Calculate the avarege training loss
  train_loss = train_loss/len(dataloader)
  train_acc = train_acc/len(dataloader)

  # Retrun train loss and accuracy of the step to track
  return train_loss, train_acc


# Creating a testing function
def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device=device):

  # Putting the model in eval mode
  model.eval()

  test_loss, test_acc = 0,0

  with torch.inference_mode():
    #
    for batch, (X, y) in enumerate(dataloader):
      X, y =  X.to(device), y.to(device)

      test_pred_logit = model(X)

      loss = loss_fn(test_pred_logit, y)

      test_loss += loss.item()

      test_pred_labels = test_pred_logit.argmax(dim=1)
      test_acc += ((test_pred_labels==y).sum().item()/len(test_pred_labels))

  test_loss /= len(dataloader)
  test_acc /= len(dataloader)
  return test_loss, test_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5,
          device = device
          ):

  # Creating a result dictionary to track the metrics
  best_model = None
  best_acc, best_loss = 0,np.inf
  results = {"train_loss":[],
             "train_acc":[],
             "test_loss":[],
             "test_acc":[]}

  # Training loo
  for epoch in tqdm(range(epochs)):
    train_loss, train_acc = train_step(model=model,
                                       dataloader=train_dataloader,
                                       loss_fn=loss_fn,
                                       optimizer=optimizer,
                                       device=device)

    test_loss, test_acc = test_step(model=model,
                                    dataloader=test_dataloader,
                                    loss_fn=loss_fn,
                                    device=device)

    # Print happening
    print(f"Epoch: {epoch} | Train loss: {train_loss:.4f}, Train Acc: {train_acc*100:.4f}% | Test loss:{test_loss:.4f}, Test acc: {test_acc*100:.4f}%")

    # Add parameters to the result dictionary
    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)

    if test_loss < best_loss:
      best_loss = test_loss
      best_model = model
      best_acc, best_loss = test_acc, test_loss

  # Return the best model as well as the  filled results dictonary
  print(f"Finished, best loss: {best_loss:.4f}, best acc: {best_acc*100:.4f}")
  return best_model,results
