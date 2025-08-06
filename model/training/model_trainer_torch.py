import sys
print(sys.version)

import os
import numpy as np
import torch
torch.manual_seed(42)

import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.optim import lr_scheduler

import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
		

# for moving data into GPU (if available)
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# for moving data to device (CPU or GPU)
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

# for loading in the device (GPU if available else CPU)
class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)
        
    def __len__(self):
        """Number of batches"""
        return len(self.dl)

class ModelTrainer:
    def __init__(self, config):
        self.img_height = config["img_height"]
        self.img_width = config["img_width"]
        self.batch_size = config["batch_size"]
        self.training_epochs = config["training_epochs"]
        self.last_epoch = self.training_epochs
        self.patience = config["patience"]
        self.data_path = config["dataset_path"]
        self.model_name = config["model_name"]
        self.model_type = config["model_type"]
        self.num_workers = config["num_workers"]
        self.requires_grad = config["requires_grad"]
        self.output_path = os.path.join(config["output_path"], self.model_name)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.class_names = None
        self.model = None
        self.history = {
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': []
        }
        # Get the versions of PyTorch and torchvision
        torch_version = torch.__version__
        torchvision_version = torchvision.__version__

        # Log the versions
        print(f"Active PyTorch version: {torch_version}")
        print(f"Active torchvision version: {torchvision_version}")
        
        self.device = get_default_device()
        print("Using device:", self.device)
        
    def load_data(self):
        transform = transforms.Compose([
            transforms.Resize((self.img_height, self.img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.train_data = datasets.ImageFolder(os.path.join(self.data_path, "train"), transform)
        self.val_data = datasets.ImageFolder(os.path.join(self.data_path, "val"), transform)
        self.test_data = datasets.ImageFolder(os.path.join(self.data_path, "test"), transform)

        self.class_names = self.train_data.classes
        print("Class names: ")
        print(self.class_names)
        print("Train dataset size:", len(self.train_data))
        print("Validation dataset size:", len(self.val_data))
        print("Test dataset size:", len(self.test_data))
    
    def preprocess_data(self):
        # Utilize PyTorch's DataLoader for batching
        self.train_loader = DeviceDataLoader(DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers), self.device)
        self.val_loader = DeviceDataLoader(DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers), self.device)
        self.test_loader = DeviceDataLoader(DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers), self.device)
    
    def convert_relu6_to_relu(self, model):
        for child_name, child in model.named_children():
            if isinstance(child, nn.ReLU6):
                inplace = child.inplace  # Get the inplace parameter, which is default=False for ReLU6
                setattr(model, child_name, nn.ReLU(inplace=inplace))
            else:
                self.convert_relu6_to_relu(child)
        # save the model to a new file
        return model
    
    def create_model(self):
        num_classes = len(self.class_names)

        if self.model_type == "shufflenet":
            self.model = torchvision.models.shufflenet_v2_x1_5(weights='DEFAULT')
            # Freeze all the layers of the pre-trained model
            for param in self.model.parameters():
                param.requires_grad = self.requires_grad
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
            # Normalize with mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
        elif self.model_type == "mobilenet":
            self.model = torchvision.models.mobilenet_v2(weights='DEFAULT')
            for param in self.model.parameters():
                param.requires_grad = self.requires_grad
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
            self.model = self.convert_relu6_to_relu(self.model)
        # EfficentNet:
        elif self.model_type == "efficientnet":
            self.model = torchvision.models.efficientnet_v2_s(weights='DEFAULT')
            for param in self.model.parameters():
                param.requires_grad = self.requires_grad
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        # densenet:
        elif self.model_type == "densenet":
            self.model = torchvision.models.densenet121(pretrained=True)
            for param in self.model.parameters():
                param.requires_grad = self.requires_grad
            self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
            # Normalize with mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
        # squeezenet:
        elif self.model_type == "squeezenet":
            self.model = torchvision.models.squeezenet1_1(weights='DEFAULT')
            for param in self.model.parameters():
                param.requires_grad = self.requires_grad
            self.model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
            # Normalize with mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
        elif self.model_type == "mnasnet":
            self.model = torchvision.models.mnasnet0_5(weights='DEFAULT')
            # self.model = torch.hub.load("coderx7/simplenet_pytorch:v1.0.0", "simplenetv1_small_m2_05", pretrained=True)
            for param in self.model.parameters():
                param.requires_grad = self.requires_grad
            self.model.classifier = nn.Linear(self.model.classifier[1].in_features, num_classes) # TODO: check if this is correct
            # Normalize with mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
        else:
            raise ValueError("Unsupported model type. Please use 'shufflenet' or 'mobilenet'.")
        # self.model = to_device(self.model, self.device) 
        self.model = self.model.to(self.device)
        
        # Check if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs for training.")
            self.model = nn.DataParallel(self.model)  # Wrap the model with DataParallel



    def train_model(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.0001)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        best_val_loss = float('inf')
        best_acc = 0.0
        self.best_epoch = 0
        patience_counter = 0

        for epoch in range(self.training_epochs):
            print(f'Epoch {epoch + 1}/{self.training_epochs}')
            self.model.train()

            running_loss = 0.0
            running_corrects = 0
            for images, labels in tqdm(self.train_loader, desc='Training', unit='batch'):
                # images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            scheduler.step()
            epoch_loss = running_loss / len(self.train_data)
            epoch_acc = running_corrects.double() / len(self.train_data)
            self.history['loss'].append(epoch_loss)
            self.history['accuracy'].append(epoch_acc.cpu())
            print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Validation phase
            val_loss, val_acc = self.validate_model()
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc.cpu())
            print(f'Validation Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

            if val_loss < best_val_loss:
                self.best_epoch = epoch
                print(f"Saving the best model at epoch {self.best_epoch+1}")
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model, os.path.join(self.output_path, self.model_name + '.pth'))
                torch.save(self.model.state_dict(), os.path.join(self.output_path, self.model_name + ".pt"))
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    self.last_epoch = epoch
                    print('Early stopping!')
                    break




    def validate_model(self):
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                # images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)
                val_loss += loss.item() * images.size(0)
                

        return val_loss / len(self.val_data), val_corrects.double() / len(self.val_data) 



    def evaluate_model(self, best_model=False):
        # load best model
        if best_model:
            print(f"Loading best model for tests from {os.path.join(self.output_path, self.model_name + '.pth')}")
            self.model.load_state_dict(torch.load(os.path.join(self.output_path, self.model_name + ".pt")))
            epoch = self.best_epoch
        else:
            epoch = self.last_epoch
        # set model in evaluation mode
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for images, labels in self.test_loader:
                # images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Test Loss: {test_loss / len(self.test_data):.4f}, Accuracy: {100 * correct / total:.2f}%")
        # save metrics of best model to csv
        test_metrics = np.array([[self.history['accuracy'][epoch], self.history['val_accuracy'][epoch], (correct / total)]])
        np.savetxt(os.path.join(self.output_path, self.model_name + "_best-metrics.csv"), test_metrics, delimiter=',', header='best_train_accuracy,best_val_accuracy,test_accuracy', comments='', fmt='%f')


    def plot_history(self):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        # Convert from tensor to numpy arrays for plotting
        train_loss = np.array(self.history['loss'])
        val_loss = np.array(self.history['val_loss'])
        train_accuracy = np.array(self.history['accuracy'])
        val_accuracy = np.array(self.history['val_accuracy'])

        # save metrics to csv
        metrics = np.array([train_loss, val_loss, train_accuracy, val_accuracy])
        print("metrics shape: ")
        print(metrics.shape)
        np.savetxt(os.path.join(self.output_path, self.model_name + "_train-metrics.csv"), metrics.T, delimiter=',', header='train_loss,val_loss,train_accuracy,val_accuracy', comments='', fmt='%f')

        ax[0].plot(train_loss, label='Training Loss')
        ax[0].plot(val_loss, label='Validation Loss')
        ax[0].set_title('Loss over epochs')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].legend()

        ax[1].plot(train_accuracy, label='Training Accuracy')
        ax[1].plot(val_accuracy, label='Validation Accuracy')
        ax[1].set_title('Accuracy over epochs')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Accuracy')
        ax[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, self.model_name + "_history.png"))
        print(f"History plot saved to {os.path.join(self.output_path, self.model_name + '_history.png')}")