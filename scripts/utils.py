"""
Contain functions to see, save and laod and so on
"""
import torch
import torchvision
from pathlib import Path
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  # Create target director
  target_dir_path = Path(target_dir)
  #target_dir_path.mkdir(parents=True, exist_ok=True)

  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with .pt or pth"

  model_save_path = target_dir_path / model_name

  # Saving state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")

  torch.save(obj=model.state_dict(),
             f=model_save_path)


def laod_model(untrained_model: torch.nn.Module,
               full_saved_dir: str,
               device:str):
  """
  Load weights of untrainded models
  """

  untrained_model.load_state_dict(torch.load(f=full_saved_dir))
  untrained_model=untrained_model.to(device)
  return untrained_model



def plot_loss_curves(results: Dict[str, List[float]]):
  """Plot lost curve recives as imput the results data dictionary
  """

  #Get the loss value of the result dictionary
  loss = results["train_loss"]
  test_loss = results["test_loss"]

  # Get the accuracy value of the results dictionary (training and test)
  accuracy =results["train_acc"]
  test_accuracy = results["test_acc"]

  # Figure out lenth
  epochs = range(len(results["train_loss"]))

  # Setup a plot
  plt.figure(figsize=(15, 7))

  # Plot loss
  plt.subplot(1, 2, 1)
  plt.plot(epochs, loss, label="train_loss")
  plt.plot(epochs, test_loss, label="test_loss")
  plt.title("Loss")
  plt.xlabel("Epochs")
  plt.legend()

  # Plot accuracy
  plt.subplot(1, 2, 2)
  plt.plot(epochs, accuracy, label="train_accuracy")
  plt.plot(epochs, test_accuracy, label="test_accuracy")
  plt.title("Accuracy")
  plt.xlabel("Epochs")
  plt.legend()


def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str,
                        class_names: List[str]=None,
                        transform=None,
                        device: torch.device = device):
  """Makse a prediction on a traget image and plots the image with its predicition"""

  # 1. Load in image and convert tensor values to float32
  target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

  # 2. (Normilise) - Study more ...Divide the image pixel values by 255 to get them between [0,1]
  target_image = target_image/255
  # or
  #target_image = np.asanyarray(target_image)

  # 3. Transform if necessary
  if transform:
    target_image = transform(target_image)

  # 5. Passing the model to device and turning evaluation mode
  model.to(device)
  model.eval()

  with torch.inference_mode():
    # Add an extra dimention to the umage (batch dimention)
    target_image = target_image.unsqueeze(dim=0)

    # Make a prediction on image with an extra dimension and send it the target device
    target_image_pred = model(target_image.to(device))

  # 6. Convert logits -> prediction probabilities
  target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

  # 7. Pred prob to label
  target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

  # 8. Plot the image
  plt.imshow(target_image.squeeze().permute(1,2,0))
  if class_names:
    title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
  else:
    title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"

  plt.title(title)
  plt.axis(False)



# Save results
def save_results(results: Dict,
                 optimizer: str,
                 lr: str,
                 model: str,
                 time: float,
                 save_dir: str=""
                 ):
  
  # Add a line in the result dictionary with time take in on cell and othres zero
  time_col = [0]*(len(results["train_acc"])-1)
  time_col.insert(0, time)
  results["time_taken"] = time_col
  
  file_name = model+"_"+optimizer+"_"+lr+".csv"
  df = pd.DataFrame(results)
  dir = save_dir+file_name
  print(f"Saving to {dir}")
  df.to_csv(dir)
