import torch


def iou_score(pred, target, threshold=0.5):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        pred = torch.tensor(pred, device=device)
        target = torch.tensor(target, device=device)
        pred = (pred > threshold).float()
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        iou = (intersection + 1e-5) / (union + 1e-5)
        # Check intersection and union
        print(f"Intersection: {intersection}, Union: {union}")
        return iou