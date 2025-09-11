"""
Creating Datalaoder for classification (recives the path of the dataset
create the dataset randomly sleet the dataset and returns the datalaoders
as well as class names)
"""
import os, torch
from  torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    dataset_dir: str,
    transform: transforms.Compose,
    batch_size: int=32,
    test_split:float=0.2,
    random_seed: int=42,
    num_workers:int=NUM_WORKERS):
  """
  Creates the datalaoders
  """

  #Creatin the dataset
  torch.manual_seed(random_seed)
  # Create the entire dataset
  full_dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)
  class_names = full_dataset.classes

  # Randomly spliting traind and test dataset
  train_size = int((1-test_split)*len(full_dataset))
  test_size = len(full_dataset) - train_size

  train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

  # Datalaoder
  train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                num_workers=num_workers, shuffle=True,
                                pin_memory=True)
  test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                               num_workers=num_workers, shuffle=False,
                               pin_memory=True)

  return train_dataloader, test_dataloader, class_names

def create_dataloaders2(
    dataset_dir: str,
    train_transform: transforms.Compose,
    test_transform: transforms.Compose,
    batch_size: int = 32,
    test_split: float = 0.2,
    random_seed: int = 42,
    num_workers: int = NUM_WORKERS):
    """
    Creates the train and test dataloaders with different transforms
    """

    # Set seed
    torch.manual_seed(random_seed)

    # Loading dataset without transform
    base_dataset = datasets.ImageFolder(root=dataset_dir)

    # Split indices
    total_size = len(base_dataset)
    train_size = int((1 - test_split) * total_size)
    test_size = total_size - train_size
    train_indices, test_indices = random_split(range(total_size), [train_size, test_size], generator=torch.Generator().manual_seed(random_seed))

    # Create full datasets again but now with transforms
    full_dataset_train = datasets.ImageFolder(root=dataset_dir, transform=train_transform)
    full_dataset_test  = datasets.ImageFolder(root=dataset_dir, transform=test_transform)

    # Use Subset to create transformed subsets
    train_dataset = Subset(full_dataset_train, train_indices)
    test_dataset = Subset(full_dataset_test, test_indices)

    # Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  num_workers=num_workers, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                 num_workers=num_workers, shuffle=False, pin_memory=True)

    class_names = full_dataset_train.classes

    return train_dataloader, test_dataloader, class_names
