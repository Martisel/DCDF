import torch
import torch.nn as nn
import torch.nn.functional as F


class TSCDLoss(nn.Module):
    """
    TSCD Loss
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, feat, fg_mask, bg_mask):
        """
        Args:
            feat: [B, C, H, W]     network feature map
            fg_mask: [B, 1, H, W]  foreground mask 
            bg_mask: [B, 1, H, W]  background mask
        Returns:
            contrastive loss
        """
        B, C, H, W = feat.shape
        N = H * W

        # L2 normalization 
        feat = F.normalize(feat, p=2, dim=1)
        feat = feat.view(B, C, N)  # [B, C, N]

        fg_mask = fg_mask.view(B, 1, N).float()
        bg_mask = bg_mask.view(B, 1, N).float()

        # Foreground & background average pooling
        sum_fg = fg_mask.sum(dim=-1, keepdim=True) + 1e-5
        sum_bg = bg_mask.sum(dim=-1, keepdim=True) + 1e-5

        f_fg = (feat * fg_mask).sum(dim=-1) / sum_fg  # [B, C]
        f_bg = (feat * bg_mask).sum(dim=-1) / sum_bg  # [B, C]

        # Concat to form contrastive queue
        f = torch.cat([f_fg, f_bg], dim=0)  # [2B, C]

        # Similarity matrix
        sim = torch.matmul(f, f.T) / self.temperature  # [2B, 2B]

        # Mask self-similarity
        self_mask = torch.eye(2 * B, device=sim.device)
        sim = sim - self_mask * 1e9

        # Labels: first B are foreground positives
        labels = torch.arange(B, device=sim.device)

        loss = F.cross_entropy(sim[:B, :], labels)

        return loss


class MultiScaleTSCDLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.loss = TSCDLoss(temperature)

    def forward(self, feat_list, fg_list, bg_list):
        total = 0.0
        for f, fg, bg in zip(feat_list, fg_list, bg_list):
            total += self.loss(f, fg, bg)
        return total / len(feat_list)
