import torch

# Better with clasification or segmentation task's if classes are unbalanced, one class is significantly outweights the other.
# Binary Focal Loss PyTorch offers a smart solution by focusing the model’s attention on the harder-to-classify examples, thereby improving overall performance.
# If gamma=0 acts as a CrossEntropyLoss, gama a hyperparameter makes loss function focus on harder to clasify examples, alpha gives more weight to positive class
def focal_loss_v3(logits, labels, gamma=1.0, alpha=0.5):
    probs = torch.sigmoid(logits)
    pt = probs * labels + (1 - probs) * (1 - labels)
    loss = -((1 - pt) ** gamma) * torch.log(pt + 1e-7)
    if alpha is not None:
        alpha_factor = alpha * labels + (1 - alpha) * (1 - labels)
        loss = alpha_factor * loss
    return loss.mean()