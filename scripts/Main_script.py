# Train Script
"""
Train a models, My main script
A version of the script is implemented on notebook on the root folder of the repo
"""
import os
import torch
import data_setup, engine, model_builder, utils
from timeit import default_timer as timer

from torchvision import transforms
device = "cuda" if torch.cuda.is_available() else "cpu"
# Setup hyperparameters
NUM_EPOCHS = 100
BATCH_SIZE = 32
TEST_SPLIT = 0.2
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

# Setup directories
data_dir = "data/EuroSAT_RGB"

# Create tranforms
transform = transforms.Compose([
    transforms.Resize((64,64)),
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(10),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Create dataloaders
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(dataset_dir="data/EuroSAT_RGB",
                                                                               transform=transform,
                                                                               batch_size=BATCH_SIZE,
                                                                               test_split = TEST_SPLIT,
                                                                               #random_seed = 42,
                                                                               #num_workers = 1
                                                                               )

## MODEL

###model = - instatiate wterever
model = None

## LOSS FUNCTION OPTMIZER

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam()

start_time = timer()
best_model, results = engine.train(model=model,
             train_datalaoder = train_datalaoder,
             test_dataloader = test_dataloader,
             loss_fn = loss_fn,
             optimizer = optimizer,
             epochs = NUM_EPOCHS,
             device=device)
end_time = timer()

time_taken = end_time-start_time

print(f"Total training time:{end_time-start_time:.3f} seconds")

#
utils.save_model(model=model,
                 target_dir = "models",
                 model_name="----.pth")

#
utils.save_results(results=results,
                   optimizer="SGD",
                   lr="0_001",
                   model="CNN_Model1",
                   time=time_taken)
