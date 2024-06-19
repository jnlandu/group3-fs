from data import load_data, train_path,val_path, get_dataframes
from resnet18_model import ResNet18
from utils import display_images, count_parameters
from args import lr,weight_decay,momentum, batch_size, num_epochs
from train import train
import torch
import torch.nn as nn

# Loading the dataset
train_df, val_df = get_dataframes()
train_df.info()
val_df.info()


### Visualizing random examples from the training and validation sets 
display_images(train_path,train_df,classes= 3, rows=1, cols=4)
display_images(val_path,val_df,classes= 3, rows=1, cols=4)

## creating the dataloaders
train_dl, val_dl = load_data(train_df, val_df,batch_size)

# loading the model
model = ResNet18()

# Inspecting number of model parameters
count_parameters(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay = weight_decay, momentum = momentum)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                 factor=0.1, patience=3, verbose=True)

#define the device to use
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
next(model.parameters()).is_cuda

## Training 
best_model, hist = train(model, num_epochs, train_dl, val_dl)

torch.save(best_model, 'best_model_0_10_epochs_ResNet18.pt')
torch.save(best_model.state_dict(), 'best_model_0_10_epochs_weights_ResNet18.pt')