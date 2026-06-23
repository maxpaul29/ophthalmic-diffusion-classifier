from typing import Sequence
import torch
import numpy as np

class Metric(torch.nn.Module):
    def __init__(self, name, device=torch.device("cpu")):
        super().__init__()
        self.name = name
        self.device = device
        self.required_output_keys = ()
        # If True, evaluate() passes continuous positive-class scores as y_pred
        self.requires_scores = False

    def reset(self):
        pass

    def update(self, output):
        pass

    def compute(self):
        pass

    def get_output(self, reduce=True):
        pass

    def set_device(self, device):
        self.device = device

    def sync_across_processes(self, accelerator):
        pass

    def __call__(self, output):
        self.update(output)
        return self.compute()
    
class Accuracy(Metric):
    def __init__(self, name, device=torch.device("cpu")):
        super().__init__(name, device)
        self.correct = torch.tensor(0)
        self.total = torch.tensor(0)
        self.device = device

    def reset(self):
        self.correct = torch.tensor(0, device=self.device)
        self.total = torch.tensor(0, device=self.device)

    def set_device(self, device):
        self.device = device
        self.correct = self.correct.to(device)
        self.total = self.total.to(device)

    def update(self, output):
        y_pred, batch = output
        y_true = batch['prompt']
        self.correct += (y_pred == y_true).sum().item()
        self.total += len(y_true)

    def sync_across_processes(self, accelerator):
        self.correct = accelerator.reduce(self.correct)
        self.total = accelerator.reduce(self.total)

    def compute(self):
        return {self.name:self.correct / self.total}

    def get_output(self, reduce=True):
        return self.compute()

class Precision(Metric):
    def __init__(self, name="precision", device=torch.device("cpu")):
        super().__init__(name, device)
        self.tp = torch.tensor(0, device=device)
        self.fp = torch.tensor(0, device=device)

    def reset(self):
        self.tp = torch.tensor(0, device=self.device)
        self.fp = torch.tensor(0, device=self.device)

    def set_device(self, device):
        self.device = device
        self.tp = self.tp.to(device)
        self.fp = self.fp.to(device)

    def update(self, output):
        """
        output: (y_pred, batch)
        y_pred: Tensor of shape [B] with 0 or 1 predicted classes
        batch["prompt"]: Tensor of shape [B] with ground truth 0 or 1
        """
        y_pred, batch = output
        y_true = batch["prompt"].to(self.device)

        # True Positives: (pred=1, true=1)
        tp_mask = (y_pred == 1) & (y_true == 1)
        # False Positives: (pred=1, true=0)
        fp_mask = (y_pred == 1) & (y_true == 0)

        self.tp += tp_mask.sum()
        self.fp += fp_mask.sum()

    def sync_across_processes(self, accelerator):
        self.tp = accelerator.reduce(self.tp)
        self.fp = accelerator.reduce(self.fp)

    def compute(self):
        tp = self.tp.float()
        fp = self.fp.float()
        denom = tp + fp
        if denom == 0:
            precision = 0.0
        else:
            precision = tp / denom
        return {self.name: precision}
    
    def get_output(self, reduce=True):
        return self.compute()


class Recall(Metric):
    def __init__(self, name="recall", device=torch.device("cpu")):
        super().__init__(name, device)
        self.tp = torch.tensor(0, device=device)
        self.fn = torch.tensor(0, device=device)

    def reset(self):
        self.tp = torch.tensor(0, device=self.device)
        self.fn = torch.tensor(0, device=self.device)

    def set_device(self, device):
        self.device = device
        self.tp = self.tp.to(device)
        self.fn = self.fn.to(device)

    def update(self, output):
        """
        output: (y_pred, batch)
        y_pred: Tensor of shape [B] with 0 or 1
        y_true: ground truth (batch["prompt"]), shape [B] with 0 or 1
        """
        y_pred, batch = output
        y_true = batch["prompt"].to(self.device)

        # True Positives: (pred=1, true=1)
        tp_mask = (y_pred == 1) & (y_true == 1)
        # False Negatives: (pred=0, true=1)
        fn_mask = (y_pred == 0) & (y_true == 1)

        self.tp += tp_mask.sum()
        self.fn += fn_mask.sum()

    def sync_across_processes(self, accelerator):
        self.tp = accelerator.reduce(self.tp)
        self.fn = accelerator.reduce(self.fn)

    def compute(self):
        tp = self.tp.float()
        fn = self.fn.float()
        denom = tp + fn
        if denom == 0:
            recall = 0.0
        else:
            recall = tp / denom
        return {self.name: recall}
    
    def get_output(self, reduce=True):
        return self.compute()

class AUC(Metric):
    """
    Unlike Accuracy/Precision/Recall/F1 -- which only need the hard predicted
    class (0/1) -- AUC needs a continuous score for the positive class to be
    meaningful (it ranks samples by confidence and measures separability across
    all thresholds). It therefore cannot be accumulated as simple counts; we
    collect all (score, label) pairs over the epoch and compute the AUC once at
    the end via the rank-based Mann-Whitney U statistic (with tie correction),
    so no extra dependency (e.g. sklearn) is required.

    This metric sets `requires_scores = True`, so `evaluate()` feeds it the
    continuous positive-class probability p(c=1|x) (softmax over the negative
    mean reconstruction errors, Eq. 3 of the paper) instead of the hard class.
    """
    def __init__(self, name="auc", device=torch.device("cpu")):
        super().__init__(name, device)
        self.requires_scores = True
        self.scores = []
        self.labels = []

    def reset(self):
        self.scores = []
        self.labels = []

    def set_device(self, device):
        self.device = device

    def update(self, output):
        """
        output: (y_pred, batch)
        y_pred: Tensor of shape [B] -- continuous score for the positive class
        y_true: ground truth (batch["prompt"]), shape [B] with 0 or 1
        """
        y_pred, batch = output
        y_true = batch["prompt"].to(self.device)

        self.scores.append(y_pred.detach().float().flatten().to(self.device))
        self.labels.append(y_true.detach().float().flatten().to(self.device))

    def sync_across_processes(self, accelerator):
        scores = torch.cat(self.scores) if self.scores else torch.empty(0, device=self.device)
        labels = torch.cat(self.labels) if self.labels else torch.empty(0, device=self.device)

        # Gather variable-length tensors from all processes for metric computation
        scores = accelerator.gather_for_metrics(scores)
        labels = accelerator.gather_for_metrics(labels)

        self.scores = [scores]
        self.labels = [labels]

    def compute(self):
        """
        Rank-based ROC-AUC (Mann-Whitney U) with tie correction:
            AUC = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        """
        if not self.scores:
            return {self.name: torch.tensor(0.0, device=self.device)}

        scores = torch.cat(self.scores).to(self.device)
        labels = torch.cat(self.labels).to(self.device)

        n_pos = (labels == 1).sum()
        n_neg = (labels == 0).sum()

        # AUC is undefined if only one class is present
        if n_pos == 0 or n_neg == 0:
            return {self.name: torch.tensor(0.0, device=self.device)}

        # Average ranks (1-indexed), averaging tied scores
        sorted_scores, order = torch.sort(scores)
        _, inverse, counts = torch.unique(sorted_scores, return_inverse=True, return_counts=True)
        # Average rank of each tie-group: end_rank - (count - 1) / 2
        end_rank = torch.cumsum(counts, dim=0).float()
        avg_rank_per_group = end_rank - (counts.float() - 1.0) / 2.0
        ranks_sorted = avg_rank_per_group[inverse]

        ranks = torch.empty_like(ranks_sorted)
        ranks[order] = ranks_sorted

        sum_ranks_pos = ranks[labels == 1].sum()
        n_pos = n_pos.float()
        n_neg = n_neg.float()

        auc = (sum_ranks_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)

        return {self.name: auc}

    def get_output(self, reduce=True):
        return self.compute()

class F1(Metric):
    def __init__(self, name="f1", device=torch.device("cpu")):
        super().__init__(name, device)
        self.tp = torch.tensor(0, device=device)
        self.fp = torch.tensor(0, device=device)
        self.fn = torch.tensor(0, device=device)

    def reset(self):
        self.tp = torch.tensor(0, device=self.device)
        self.fp = torch.tensor(0, device=self.device)
        self.fn = torch.tensor(0, device=self.device)

    def set_device(self, device):
        self.device = device
        self.tp = self.tp.to(device)
        self.fp = self.fp.to(device)
        self.fn = self.fn.to(device)

    def update(self, output):
        """
        output: (y_pred, batch)
        y_pred: Tensor of shape [B] with 0 or 1
        y_true: ground truth (batch["prompt"]), shape [B] with 0 or 1
        """
        y_pred, batch = output
        y_true = batch["prompt"].to(self.device)

        # True Positives
        tp_mask = (y_pred == 1) & (y_true == 1)
        # False Positives
        fp_mask = (y_pred == 1) & (y_true == 0)
        # False Negatives
        fn_mask = (y_pred == 0) & (y_true == 1)

        self.tp += tp_mask.sum()
        self.fp += fp_mask.sum()
        self.fn += fn_mask.sum()

    def sync_across_processes(self, accelerator):
        self.tp = accelerator.reduce(self.tp)
        self.fp = accelerator.reduce(self.fp)
        self.fn = accelerator.reduce(self.fn)

    def compute(self):
        """
        F1 = 2 * TP / (2*TP + FP + FN)
        """
        tp = self.tp.float()
        fp = self.fp.float()
        fn = self.fn.float()

        numerator = 2.0 * tp
        denominator = numerator + fp + fn

        if denominator == 0:
            f1 = 0.0
        else:
            f1 = numerator / denominator

        return {self.name: f1}
    
    def get_output(self, reduce=True):
        return self.compute()
