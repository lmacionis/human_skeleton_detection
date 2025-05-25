import torch
import torchvision
from torchvision import transforms
from torchsummary import summary
from check_data import *
import torch.nn as nn
import torch.optim as optim
from unet_decoder import UNet
from torchvision.models import resnet50
from dataset_custom import CustomDataset
from save_model import save_model
from intersection_over_union import iou_score
import numpy as np
from torchvision.models.feature_extraction import create_feature_extractor
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

mean = [0.6981, 0.6837, 0.6703]
std = [0.3245, 0.3306, 0.3320]


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


train_dataset = CustomDataset(path_train, path_train_coord, transform=transform)
test_dataset = CustomDataset(path_test, path_test_coord, transform=transform)
valid_dataset = CustomDataset(path_valid, path_valid_coord, transform=transform)

# Trained on batch size 16
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)   # num_workers -> speed up trainin process by the use of cpu, size depends from cpu cores
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=8, shuffle=True)

# train_iamge_shape, classification_map = next(iter(train_dataloader))       # takes only one image at a time and unpacks into variables
# # For testing: image after transformations with clasification map / ground truth mask
# print(f"Train image shape after transformations: {train_iamge_shape.size()}")
# print(f"Ground truth mask: {classification_map.size()}")
# img = train_iamge_shape[0].squeeze()
# img = np.mean(img.numpy(), axis=0)
# plt.imshow(img)
# plt.show()


resnet = resnet50(weights=True)
feature_extractor = create_feature_extractor(resnet, {'layer4': 'layer4'})      # Extracting layer4 feautre map, before Fully Connected layer.
feature_extractor.eval()


unet_model = UNet(in_channels=2048)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet_model = unet_model.to(device)
feature_extractor = feature_extractor.to(device)
summary(unet_model, (2048, 20, 20), device=device.type)


# criterion = focal_loss_v3
# Focal loss gets stuck on predicting zero values, had to use BCEWithLogitsLoss, it has sigmoid function.
# Also more stable then counting focal loss and using sigmoid in decoder, pos_wight tells to focus on positive pixells, have to use it cause of high class imbalance
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([50.0]).to(device))
# criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(unet_model.parameters(), lr=0.0001)

# trained on 20 epochs
def train_model(unet_model, train_dataloader, valid_dataloader, criterion, optimizer, epochs=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Naudojamas įrenginys: {device}")

    unet_model = unet_model.to(device)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        unet_model.train()
        train_loss = 0
        correct = 0
        total = 0

        for images, classification_map in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, classification_map = images.to(device), classification_map.to(device)
            classification_map = classification_map.unsqueeze(1)
            # print(classification_map.shape)
            # print(images.shape)

            features = feature_extractor(images)["layer4"]
            optimizer.zero_grad()
            outputs_image = unet_model(features)
            # print(outputs_image.shape)
            # print(classification_map.shape)

            loss = criterion(outputs_image, classification_map)
            loss.backward()

            optimizer.step()
            train_loss += loss.item()
            predicted = (outputs_image > 0.5).float()   # 0.5 takes values above as good below as bad

            correct += (predicted == classification_map).sum().item()
            total += classification_map.numel()

        train_losses.append(train_loss / len(train_dataloader))
        train_accuracies.append(100 * correct / total)

        unet_model.eval()

        val_loss = 0
        correct = 0
        total = 0

        for images, classification_map in valid_dataloader:
            images, classification_map = images.to(device), classification_map.to(device)
            classification_map = classification_map.unsqueeze(1)

            with torch.no_grad():
                features = feature_extractor(images)["layer4"]
                outputs_image = unet_model(features)

            loss = criterion(outputs_image, classification_map)
            val_loss += loss.item()
            predicted = (outputs_image > 0.5).float()

            correct += (predicted == classification_map).sum().item()
            total += classification_map.numel()

        val_losses.append(val_loss / len(valid_dataloader))
        val_accuracies.append(100 * correct / total)

        print(f"Epocha {epoch + 1} / {epochs}")
        print(f"   Treniravimo Nuostolis: {train_losses[-1]:.4f}, Tikslumas: {train_accuracies[-1]:.2f}%")
        print(f"   Validavimo Nuostolis: {val_losses[-1]:.4f},  Tikslumas: {val_accuracies[-1]:.2f}%")

    return train_losses, train_accuracies, val_losses, val_accuracies

train_losses, train_accuracies, val_losses, val_accuracies = train_model(unet_model, train_dataloader, valid_dataloader, criterion, optimizer)
save_model(unet_model=unet_model, optimizer=optimizer, epoch=20, train_losses=train_losses, train_accuracies=train_accuracies, val_losses=val_losses, val_accuracies=val_accuracies, path="./model//model.pth")


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Treniravimo nuostolis")
plt.plot(val_losses, label="Validacijos nuostolis")
plt.xlabel("Epocha")
plt.ylabel("Nuostolis")
plt.title("Nuostolio kaita per epochas")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Treniravimo tikslumas")
plt.plot(val_accuracies, label="Validacijos tikslumas")
plt.xlabel("Epocha")
plt.ylabel("Tikslumas (%)")
plt.title("Tikslumo kaita per epochas")
plt.legend()

plt.tight_layout()
plt.show()

def test_model(unet_model, test_loader):
    unet_model.eval()
    correct = 0
    total = 0
    true_clasification_map = []
    predicted_map = []
    iou_scores = []

    with torch.no_grad():
        for batch_idx, (images, classification_map) in enumerate(tqdm(test_loader, desc="Testing")):
            images, classification_map = images.to(device), classification_map.to(device)
            classification_map = classification_map.unsqueeze(1)

            features = feature_extractor(images)["layer4"]
            outputs_image = unet_model(features)
            predicted = (outputs_image > 0.1).float()

            batch_iou = iou_score(predicted, classification_map)
            mean_batch_iou = batch_iou.mean().item()
            iou_scores.append(mean_batch_iou)
            print(f"Batch {batch_idx + 1}/{len(test_loader)} - Mean IoU: {mean_batch_iou:.4f}")

            predicted_map.extend(predicted.cpu().numpy().flatten())
            true_clasification_map.extend(classification_map.cpu().numpy().flatten())

            correct += (predicted == classification_map).sum().item()
            total += classification_map.numel()


    predicted_map = np.round(predicted_map).astype(int)
    true_clasification_map = np.round(true_clasification_map).astype(int)

    precision = precision_score(true_clasification_map, predicted_map)
    recall = recall_score(true_clasification_map, predicted_map)
    f1 = f1_score(true_clasification_map, predicted_map)
    iuo = iou_score(predicted_map, true_clasification_map)
    mean_iou = np.mean(iou_scores)

    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}, Mean IOU: {mean_iou:.2f}, Accuracy: {(100 * correct / total):.2f}%")

    # Visualize a sample prediction
    images, classification_map = next(iter(test_loader))
    images, classification_map = images.to(device), classification_map.to(device)
    with torch.no_grad():
        features = feature_extractor(images)["layer4"]
        outputs_image = unet_model(features)
        predicted = (outputs_image > 0.1).float()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    # Selects first image, reorders to fit matplotlib, puts in cpu since np arrays does not work on gpu, creates numpy arrays and dnormalize to use as an image 
    img = images[0].permute(1, 2, 0).cpu().numpy() * np.array(std) + np.array(mean)
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.title("Input Image")
    plt.subplot(1, 2, 2)
    plt.imshow(predicted[0, 0].cpu().numpy(), cmap="gray")
    plt.title("Predicted Mask")
    plt.savefig('sample_prediction.png')
    plt.close()

    return predicted_map, true_clasification_map

predicted_map, true_clasification_map = test_model(unet_model, test_dataloader)
cm = confusion_matrix(true_clasification_map, predicted_map)
plt.figure(figsize=(5, 5))
sns.heatmap(cm / cm.sum(axis=1, keepdims=True), annot=True, fmt=".2f", cmap="Blues")    # Normalizing confusion matrix for better visualization
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Real")
plt.show()