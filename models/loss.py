import torch
import torch.nn as nn


class WeightedLoss(nn.Module):

    def __init__(self):
        super(WeightedLoss, self).__init__()
        self.weighted = False

    def generate_weight_mask(self, mask, to_ignore=None):
        """ Generates a weight mask where pixel weights are inversely proportional to
            how many pixels are in the class
            @param mask: a [N x ...] torch.FloatTensor with values in {0, 1, 2, ..., K+1}, where K is number of objects. {0,1} are background/table.
            @param to_ignore: a list of classes (integers) to ignore when creating mask
            @return: a torch.FloatTensor that is same shape as mask.
        """
        N = mask.shape[0]

        if self.weighted:

            # Compute pixel weights
            weight_mask = torch.zeros_like(mask).float() # Shape: [N x H x W]. weighted mean over pixels

            for i in range(N):

                unique_object_labels = torch.unique(mask[i])
                for obj in unique_object_labels: # e.g. [0, 1, 2, 5, 9, 10]. bg, table, 4 objects

                    if to_ignore is not None and obj in to_ignore:
                        continue

                    num_pixels = torch.sum(mask[i] == obj, dtype=torch.float)
                    weight_mask[i, mask[i] == obj] = 1 / num_pixels # inversely proportional to number of pixels

        else:
            weight_mask = torch.ones_like(mask) # mean over observed pixels
            if to_ignore is not None:
                for obj in to_ignore:
                    weight_mask[mask == obj] = 0

        return weight_mask

class BCEWithLogitsLossWeighted(WeightedLoss):
    """ Compute weighted BCE loss with logits
    """
    def __init__(self, weighted=False):
        super(BCEWithLogitsLossWeighted, self).__init__()
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduction='none')
        self.weighted = weighted

    def forward(self, x, target):
        """ Compute masked cosine similarity loss
            @param x: a [N x H x W] torch.FloatTensor of foreground logits
            @param target: a [N x H x W] torch.FloatTensor of values in [0, 1]
        """
        temp = self.BCEWithLogitsLoss(x, target) # Shape: [N x H x W]. values are in [0, 1]
        weight_mask = self.generate_weight_mask(target)
        loss = torch.sum(temp * weight_mask, (1, 2, 3)) / torch.sum(weight_mask, (1, 2, 3))

        return loss
