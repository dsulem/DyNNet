import torch

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function based on Euclidean distance (y=1 means similar, y=0 means dissimilar)
    """

    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self,distance, y):
        loss = (1 - y) * torch.clamp(self.margin - distance, min=0.0) + y * distance
        loss = torch.sum(loss) / 2.0 / loss.size()[0]
        return loss.reshape(1)


class HingeLoss(torch.nn.Module):
    """
    Hinge loss function based on similarity metric (y=1 means similar, y=0 means dissimilar)
    """

    def __init__(self, margin=.5, weight=1.0):
        super(HingeLoss, self).__init__()
        self.margin = margin
        self.weight = weight

    def forward(self, similarity, y):
        loss = (1 - y) * self.weight * torch.clamp(similarity + self.margin, min=0.0) + y * torch.clamp(self.margin - similarity, min=0.0)
        loss = torch.sum(loss) / 2.0 / similarity.size()[0]
        return loss.reshape(1)