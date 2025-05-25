import torch.nn.functional as F

def dice_loss(pred, target, smooth=1):
    # print(pred.shape)
    # print(target.shape)
    pred = F.softmax(pred, dim=1)
    num_classes = pred.shape[1]
    dice = 0
    for c in range(num_classes):
        pred_c = pred[:, c]
        target_c = target[:, c]
        intersection = (pred_c * target_c).sum(dim=(1, 2))
        union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))
        dice += (2. * intersection + smooth) / (union + smooth)

    return 1 - dice.mean() / num_classes