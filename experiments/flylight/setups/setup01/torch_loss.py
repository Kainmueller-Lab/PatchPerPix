"""Provides a wrapper class for the losses used
"""
import math
import sys

import numpy as np
import torch
import torchmetrics



class MaskedCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    def __init__(self, *args, num_channels=None, use_gt_extra=False, **kwargs):
        assert kwargs.get('reduction') == "none", (
            "use 'none' reduction for masked loss!")
        super().__init__(*args, **kwargs)
        self.num_channels = num_channels
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.use_gt_extra = use_gt_extra

    def forward(self, input, target, mask):
        if mask is not None:
            if self.use_gt_extra:
                out = self.log_softmax(input)
                target = torch.nn.functional.one_hot(target, num_classes=3)
                target = torch.movedim(target, -1, 1)
                loss = - (out * target)
                cnt = torch.sum(mask)
                t = torch.sum(mask, dim=1)
            else:
                loss = super().forward(input, target)
                if mask is not None:
                    cnt = (torch.sum(mask)*self.num_channels)
            if cnt == 0:
                loss = torch.mean(loss) * 0.0
            else:
                loss = torch.sum((loss*mask)) / cnt
        else:
            loss = super().forward(input, target)
            if loss.numel() == 0:
                loss = loss.sum()
            else:
                loss = loss.mean()
        return loss


class MaskedBCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    def __init__(self, *args, num_channels=None, **kwargs):
        assert kwargs.get('reduction') == "none", (
            "use 'none' reduction for masked loss!")
        super().__init__(*args, **kwargs)
        self.num_channels = num_channels

    def forward(self, input, target, mask):
        loss = super().forward(input, target)
        if mask is not None:
            ch = 0 if mask.size(1) == 1 else 1
            cnt = (torch.sum(mask[:,ch,...])*self.num_channels)
            if cnt == 0:
                loss = torch.mean(loss) * 0.0
            else:
                loss = torch.sum((loss*mask[:,ch,...].unsqueeze_(1))) / cnt
        elif loss.numel() == 0:
            loss = loss.sum()
        else:
            loss = loss.mean()
        return loss


class LossWrapper(torch.nn.Module):
    """Wraps a set of torch losses and tensorboard summaries used to train
    the tracking model of Linajea
    """
    def __init__(self, config, current_step=0):
        super().__init__()
        self.config = config
        self.current_step = torch.nn.Parameter(
            torch.tensor(float(current_step)), requires_grad=False)

        self.overlapping_inst = self.config.get('overlapping_inst')
        self.train_code = self.config.get("train_code")
        self.patchshape_squeezed = tuple(p for p in self.config['patchshape']
                                         if p > 1)
        self.patchsize = int(np.prod(self.patchshape_squeezed))

        self.patch_loss = MaskedBCEWithLogitsLoss(
            reduction="none", num_channels=float(self.patchsize))

        if self.overlapping_inst:
            self.fg_numinst_loss = MaskedCrossEntropyLoss(
                reduction="none", num_channels=1.0,
                use_gt_extra=config.get("use_gt_extra"))
        else:
            self.fg_numinst_loss = MaskedBCEWithLogitsLoss(
                reduction="none", num_channels=1.0,
                use_gt_extra=config.get("use_gt_extra"))


        self.jaccard = torchmetrics.classification.BinaryJaccardIndex()
        self.accuracy = torchmetrics.classification.BinaryAccuracy()
        self.accuracy2 = torchmetrics.classification.BinaryAccuracy(ignore_index=0)
        self.mse = torchmetrics.MeanSquaredError()
        met_sum_intv = 10
        loss_sum_intv = 1
        self.summaries = {
            "loss":                            [-1, loss_sum_intv],
            "loss_patch":                      [-1, loss_sum_intv],
            "loss_fg":                         [-1, loss_sum_intv],
            "jaccard_fg":                         [-1, loss_sum_intv],
            "accuracy_fg":                        [-1, loss_sum_intv],
            "accuracy2_fg":                       [-1, loss_sum_intv],
            "mse_fg":                       [-1, loss_sum_intv],
            "jaccard_patch":                         [-1, loss_sum_intv],
            "accuracy_patch":                        [-1, loss_sum_intv],
            "accuracy2_patch":                       [-1, loss_sum_intv],
            "mse_patch":                       [-1, loss_sum_intv],
            }

    def forward(
            self, *,
            pred_logits_affs,
            pred_logits_fg,
            gt_affs_samples,
            gt_fgbg_loss,
            loss_mask=None
    ):
        loss_patch = self.patch_loss(
            pred_logits_affs, gt_affs_samples,
            loss_mask if not self.train_code and loss_mask is not None else None)
        loss_fg = self.fg_numinst_loss(pred_logits_fg, gt_fgbg_loss, loss_mask)
        loss = loss_patch + loss_fg

        if pred_logits_affs.numel() > 0:
            self.summaries['jaccard_patch'][0] = self.jaccard(torch.flatten(
                pred_logits_affs), torch.flatten(gt_affs_samples))
            self.summaries['accuracy_patch'][0] = self.accuracy(torch.flatten(
                pred_logits_affs), torch.flatten(gt_affs_samples))
            self.summaries['accuracy2_patch'][0] = self.accuracy2(torch.flatten(
                pred_logits_affs), torch.flatten(gt_affs_samples))
            self.summaries['mse_patch'][0] = self.mse(torch.flatten(
                torch.sigmoid(pred_logits_affs)), torch.flatten(gt_affs_samples))

        if self.overlapping_inst:
            gt_fgbg_loss = torch.nn.functional.one_hot(
                gt_fgbg_loss, num_classes=self.config['max_num_inst']+1)
            gt_fgbg_loss = torch.movedim(gt_fgbg_loss, -1, 1)
            pred_logits_fg = torch.softmax(pred_logits_fg, dim=1)
        else:
            pred_logits_fg = torch.sigmoid(pred_logits_fg)

        self.summaries['jaccard_fg'][0] = self.jaccard(torch.flatten(
            pred_logits_fg), torch.flatten(gt_fgbg_loss))
        self.summaries['accuracy_fg'][0] = self.accuracy(torch.flatten(
            pred_logits_fg), torch.flatten(gt_fgbg_loss))
        self.summaries['accuracy2_fg'][0] = self.accuracy2(torch.flatten(
            pred_logits_fg), torch.flatten(gt_fgbg_loss))
        self.summaries['mse_fg'][0] = self.mse(torch.flatten(
            pred_logits_fg), torch.flatten(gt_fgbg_loss))

        self.summaries['loss'][0] = loss
        self.summaries['loss_patch'][0] = loss_patch
        self.summaries['loss_fg'][0] = loss_fg


        self.current_step += 1
        outputs = {
            "loss": loss,
            "summaries": self.summaries,
        }
        return outputs
