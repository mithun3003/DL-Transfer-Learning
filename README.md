# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM

To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## Problem Statement and Dataset

### Problem Statement:

To develop a deep learning classification model that accurately identifies and classifies chip images into their respective categories using transfer learning. The pretrained VGG19 model is used for feature extraction, and its final fully connected layer is modified according to the number of classes in the dataset. The model is trained and evaluated using accuracy, confusion matrix, precision, recall, and F1-score.

### Dataset:
- Dataset Name: Chip Image Dataset (chip_data.zip)
- Dataset Type: Image classification

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:
Load and Preprocess Dataset
Import the required libraries, extract the chip dataset, resize images to 224 × 224 pixels, normalize them, and load training and testing images using ImageFolder and DataLoader.

### STEP 2: 
Load Pretrained VGG19 Model
Load the VGG19 model with pretrained ImageNet weights and freeze the convolutional feature extraction layers to preserve learned features.

### STEP 3: 
Modify the Classification Layer
Replace the final fully connected layer of VGG19 with a new linear layer corresponding to the number of chip dataset classes.

### STEP 4: Train the Model

Define CrossEntropyLoss and Adam optimizer. Train the model for 10 epochs using forward propagation, loss calculation, backpropagation, and weight updates.

### STEP 5: 

Evaluate the Model
Evaluate the trained model using test accuracy, training and validation loss curves, confusion matrix, and classification report containing precision, recall, and F1-score.

### STEP 6: 
Predict New Images
Select sample images from the testing dataset, predict their classes using the trained VGG19 model, and display the actual and predicted labels.


## PROGRAM

### Name: MITHUN S

### Register Number: 212224240088

```python

# ============================================================
# DL - DEVELOPING A NEURAL NETWORK CLASSIFICATION MODEL
# USING TRANSFER LEARNING - VGG19
# ============================================================

# STUDENT DETAILS

NAME = "MITHUN S"
REGISTER_NUMBER = "212224240088"

print("======================================")
print("NAME:", NAME)
print("REGISTER NUMBER:", REGISTER_NUMBER)
print("======================================")


# ============================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================

import os
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import models, datasets
from torchvision.models import VGG19_Weights

from torchsummary import summary

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# ============================================================
# STEP 1: LOAD AND PREPROCESS DATA
# ============================================================

# Extract Dataset

with zipfile.ZipFile("./chip_data.zip", "r") as zip_ref:
    zip_ref.extractall("./data")


# Define Image Transformations

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# Load Dataset

dataset_path = "./data/dataset/"

train_dataset = datasets.ImageFolder(
    root=f"{dataset_path}/train",
    transform=transform
)

test_dataset = datasets.ImageFolder(
    root=f"{dataset_path}/test",
    transform=transform
)


# Get Class Names

class_names = train_dataset.classes
num_classes = len(class_names)

assert train_dataset.class_to_idx == test_dataset.class_to_idx, \
    "Training and testing classes must match."


# Display Dataset Details

print("\n========== DATASET DETAILS ==========")
print("Name:", NAME)
print("Register Number:", REGISTER_NUMBER)
print("Training Samples:", len(train_dataset))
print("Testing Samples:", len(test_dataset))
print("Classes:", class_names)
print("Number of Classes:", num_classes)


# Display Sample Images

def show_sample_images(dataset, num_images=5):

    num_images = min(num_images, len(dataset))

    fig, axes = plt.subplots(
        1, num_images, figsize=(12, 4)
    )

    axes = np.atleast_1d(axes)

    for i in range(num_images):

        image, label = dataset[i]

        image = image.permute(1, 2, 0).numpy()

        # Reverse Normalization

        image = image * np.array(
            [0.229, 0.224, 0.225]
        )

        image = image + np.array(
            [0.485, 0.456, 0.406]
        )

        image = np.clip(image, 0, 1)

        axes[i].imshow(image)
        axes[i].set_title(class_names[label])
        axes[i].axis("off")

    plt.suptitle(
        f"Name: {NAME} | Register No: {REGISTER_NUMBER}"
    )

    plt.tight_layout()
    plt.show()


show_sample_images(train_dataset)


# Display Image Shape

first_image, label = train_dataset[0]

print("Shape of First Training Image:", first_image.shape)

test_image, label = test_dataset[0]

print("Shape of First Testing Image:", test_image.shape)


# Create DataLoaders

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


# ============================================================
# STEP 2: LOAD PRETRAINED VGG19 MODEL
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\n========== MODEL DETAILS ==========")
print("Name:", NAME)
print("Register Number:", REGISTER_NUMBER)
print("Device:", device)


# Load Pretrained VGG19 Model

model = models.vgg19(
    weights=VGG19_Weights.DEFAULT
)


# Freeze Feature Extraction Layers

for param in model.features.parameters():
    param.requires_grad = False


# Freeze Original Classifier Layers

for param in model.classifier.parameters():
    param.requires_grad = False


# Modify Final Fully Connected Layer

model.classifier[6] = nn.Linear(
    4096, num_classes
)


# Move Model to GPU or CPU

model = model.to(device)


# Print Model Summary

print("\n========== VGG19 MODEL SUMMARY ==========")
print("Name:", NAME)
print("Register Number:", REGISTER_NUMBER)

summary(
    model,
    input_size=(3, 224, 224),
    device=str(device)
)


# ============================================================
# STEP 3: LOSS FUNCTION AND OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.classifier[6].parameters(),
    lr=0.001
)

print("\n========== TRAINING CONFIGURATION ==========")
print("Name:", NAME)
print("Register Number:", REGISTER_NUMBER)
print("Loss Function: CrossEntropyLoss")
print("Optimizer: Adam")
print("Learning Rate: 0.001")
print("Batch Size: 32")


# ============================================================
# STEP 4: TRAIN THE MODEL
# ============================================================

def train_model(
    model,
    train_loader,
    test_loader,
    num_epochs=10
):

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):

        model.train()

        running_loss = 0.0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Clear Gradients

            optimizer.zero_grad()

            # Forward Propagation

            outputs = model(images)

            # Calculate Loss

            loss = criterion(outputs, labels)

            # Backpropagation

            loss.backward()

            # Update Weights

            optimizer.step()

            running_loss += (
                loss.item() * images.size(0)
            )

        epoch_train_loss = (
            running_loss / len(train_loader.dataset)
        )

        train_losses.append(epoch_train_loss)


        # Validation Loss

        model.eval()

        validation_loss = 0.0

        with torch.no_grad():

            for images, labels in test_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)

                validation_loss += (
                    loss.item() * images.size(0)
                )

        epoch_val_loss = (
            validation_loss / len(test_loader.dataset)
        )

        val_losses.append(epoch_val_loss)


        # Display Epoch Results

        print("\nName:", NAME)
        print("Register Number:", REGISTER_NUMBER)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Train Loss: {epoch_train_loss:.4f}, "
            f"Validation Loss: {epoch_val_loss:.4f}"
        )


    # Plot Training and Validation Loss

    plt.figure(figsize=(8, 6))

    plt.plot(
        range(1, num_epochs + 1),
        train_losses,
        label="Training Loss",
        marker="o"
    )

    plt.plot(
        range(1, num_epochs + 1),
        val_losses,
        label="Validation Loss",
        marker="s"
    )

    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.title(
        "Training and Validation Loss\n"
        f"{NAME} - {REGISTER_NUMBER}"
    )

    plt.legend()
    plt.grid(True)
    plt.show()

    return train_losses, val_losses


# Train Model

train_losses, val_losses = train_model(
    model,
    train_loader,
    test_loader,
    num_epochs=10
)


# ============================================================
# STEP 5: TEST MODEL AND COMPUTE CONFUSION MATRIX
# ============================================================

def test_model(model, test_loader):

    model.eval()

    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

            all_preds.extend(
                predicted.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )


    # Calculate Accuracy

    accuracy = correct / total

    print("\n========== MODEL EVALUATION ==========")
    print("Name:", NAME)
    print("Register Number:", REGISTER_NUMBER)

    print("Test Accuracy:", round(accuracy, 4))

    print(
        "Test Accuracy Percentage:",
        round(accuracy * 100, 2), "%"
    )


    # Compute Confusion Matrix

    cm = confusion_matrix(
        all_labels,
        all_preds,
        labels=list(range(num_classes))
    )

    print("\n========== CONFUSION MATRIX ==========")
    print("Name:", NAME)
    print("Register Number:", REGISTER_NUMBER)

    print(cm)


    # Plot Confusion Matrix

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.title(
        "Confusion Matrix - VGG19\n"
        f"{NAME} - {REGISTER_NUMBER}"
    )

    plt.tight_layout()
    plt.show()


    # Classification Report

    print("\n========== CLASSIFICATION REPORT ==========")
    print("Name:", NAME)
    print("Register Number:", REGISTER_NUMBER)

    report = classification_report(
        all_labels,
        all_preds,
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0
    )

    print(report)

    return accuracy


# Evaluate Model

test_accuracy = test_model(
    model,
    test_loader
)


# ============================================================
# STEP 6: PREDICT A SINGLE IMAGE
# ============================================================

def predict_image(model, image_index, dataset):

    model.eval()

    if image_index < 0 or image_index >= len(dataset):

        print("Invalid image index")
        return

    image, label = dataset[image_index]

    with torch.no_grad():

        image_tensor = image.unsqueeze(0).to(device)

        output = model(image_tensor)

        # Softmax Probability

        prob = torch.softmax(output, dim=1)

        # Get Predicted Class

        predicted = torch.argmax(
            prob, dim=1
        ).item()

        confidence = prob[0, predicted].item()


    # Reverse Normalization

    image_to_display = image.permute(
        1, 2, 0
    ).numpy()

    image_to_display = (
        image_to_display *
        np.array([0.229, 0.224, 0.225])
    )

    image_to_display = (
        image_to_display +
        np.array([0.485, 0.456, 0.225])
    )

    image_to_display = np.clip(
        image_to_display, 0, 1
    )


    # Display Image

    plt.figure(figsize=(5, 5))

    plt.imshow(image_to_display)

    plt.title(
        f"Name: {NAME}\n"
        f"Register No: {REGISTER_NUMBER}\n"
        f"Actual: {class_names[label]}\n"
        f"Predicted: {class_names[predicted]}\n"
        f"Confidence: {confidence*100:.2f}%"
    )

    plt.axis("off")
    plt.show()


    # Display Prediction Details

    print("\n========== SAMPLE IMAGE PREDICTION ==========")

    print("Name:", NAME)
    print("Register Number:", REGISTER_NUMBER)

    print("Actual:", class_names[label])

    print("Predicted:", class_names[predicted])

    print(
        "Confidence:",
        round(confidence * 100, 2), "%"
    )


# Example Prediction 1

predict_image(
    model,
    image_index=55,
    dataset=test_dataset
)


# Example Prediction 2

predict_image(
    model,
    image_index=25,
    dataset=test_dataset
)


# ============================================================
# SAVE TRAINED MODEL
# ============================================================

torch.save(
    model.state_dict(),
    "vgg19_chip_model.pth"
)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n============================================")
print("FINAL RESULT")
print("============================================")

print("Name:", NAME)
print("Register Number:", REGISTER_NUMBER)

print("Model: VGG19 Transfer Learning")

print("Dataset: Chip Image Dataset")

print("Number of Classes:", num_classes)

print(
    "Final Test Accuracy:",
    round(test_accuracy * 100, 2), "%"
)

print("Training Epochs: 10")

print("Model Saved: vgg19_chip_model.pth")

print("Image Classification Completed Successfully!")

print("============================================")



```

### OUTPUT

## Training Loss, Validation Loss Vs Iteration Plot

<img width="585" height="1160" alt="image" src="https://github.com/user-attachments/assets/cad6bbd4-1e15-47fe-98e1-ae84b22c2cd2" />

## Confusion Matrix

<img width="632" height="582" alt="image" src="https://github.com/user-attachments/assets/e6087723-9c45-4c5c-9b37-ca09248d8aa6" />

## Classification Report

<img width="478" height="610" alt="image" src="https://github.com/user-attachments/assets/2d9c2836-b2fd-4923-b9f8-51cbab0f0911" />

### New Sample Data Prediction

<img width="407" height="625" alt="image" src="https://github.com/user-attachments/assets/427e7683-38d9-4700-a9f9-d322825e7dd5" />

## RESULT
Thus, the image classification model using transfer learning with the pretrained VGG19 architecture was successfully developed using PyTorch. The model was trained for 10 epochs on the chip image dataset, and its performance was evaluated using training loss, validation loss, accuracy, confusion matrix, precision, recall, and F1-score. The model was also used to predict the classes of sample test images. The experiment demonstrated the application of transfer learning for image classification using pretrained deep learning models.
