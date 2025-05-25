import torch

def save_model(unet_model, optimizer, epoch, train_losses, train_accuracies, val_losses, val_accuracies, path="./model//model.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': unet_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }, path)
    print(f"Model saved to {path}")