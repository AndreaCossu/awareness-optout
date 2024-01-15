import torch
import numpy as np


class EarlyStopping:
    def __init__(self, mode='min', patience=3):
        self.stop_count = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.patience = patience
        self.mode = mode
        self.stopped = False

    def update(self, current_value):

        if self.mode == 'min':
            no_improve = current_value >= self.best_value
        else:
            no_improve = current_value <= self.best_value

        if no_improve:
            self.stop_count += 1
        else:
            self.best_value = current_value

        self.stopped = self.stop_count == self.patience
        return self.stopped

    def reset(self):
        self.stopped = False
        self.stop_count = 0
        self.best_value = float('inf') if self.mode == 'min' else float('-inf')


class GamblersLoss:
    def __init__(self, temperature):
        """
        https://proceedings.neurips.cc/paper/2019/hash/0c4b1eeb45c90b52bfb9d07943d855ab-Abstract.html
        :param temperature: >1, <= number of classes (o in the original paper)
        """
        self.temperature = temperature

    def __call__(self, out, target):
        probs = torch.softmax(out, dim=-1)
        outputs, reservation = probs[:, :-1], probs[:, -1]
        gain = torch.gather(outputs, dim=1, index=target.unsqueeze(1)).squeeze()
        doubling_rate = (gain + (reservation / self.temperature)).log()
        return -doubling_rate.mean()


@torch.no_grad()
def compute_example_confidences(method, out):
    if method == 'none':
        confidences = torch.softmax(out, dim=-1)
        confidences = confidences.max(dim=-1)[0]
    elif method == 'gamblers':
        confidences = torch.softmax(out, dim=-1)
        # gamblers value is confidence to reject, therefore 1 - to get confidence to predict
        confidences = 1 - confidences[:, -1]
    else:
        raise ValueError("Wrong method name.")

    return confidences


@torch.no_grad()
def compute_dataset_confidences_predictions(mdl, method, loader, loss_fn, device):
    mdl.eval()
    confidences = []
    predictions = []
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = mdl(x)
        cfd = compute_example_confidences(method, out)
        confidences.append(cfd)

        for i in range(x.size(0)):
            losses.append(loss_fn(out[i].unsqueeze(0), y[i].unsqueeze(0)))

        if method == 'gamblers':
            out = out[:, :-1]
        predictions.append(out.argmax(dim=-1) == y)

    confidences = torch.cat(confidences, dim=0)
    predictions = torch.cat(predictions, dim=0).float()
    losses = torch.tensor(losses).to(confidences.device)
    return confidences, predictions, losses


@torch.no_grad()
def evaluate_with_threshold(confidences, predictions, losses, confidence_threshold=0.0):
    mask = confidences >= confidence_threshold
    num_predictions = mask.sum().item()
    if num_predictions == 0:
        return 1.0, 0, 0

    correct = predictions[mask].sum().item()
    acc = 100 * (correct / float(num_predictions))
    loss = losses[mask].mean().item()

    return acc, loss, num_predictions


@torch.no_grad()
def evaluate_coverage(confidences, predictions, coverages):
    confidences = confidences.cpu().numpy().tolist()
    predictions = predictions.cpu().numpy().tolist()
    sorted_accs = sorted([(c, p) for c, p in zip(confidences, predictions)], key=lambda el: el[0], reverse=True)

    coverages_accs = {}
    for c in coverages:
        num_elements = int(float(c/100) * len(sorted_accs))
        coveraged = sorted_accs[:num_elements]
        coverages_accs[c] = 100*(sum([el[1] for el in coveraged]) / num_elements)

    return coverages_accs


@torch.no_grad()
def evaluate_calibration(confidences, predictions, bins):
    ece = {}

    max_bin = 1.0
    for min_bin in bins:
        mask = torch.logical_and(confidences <= max_bin, confidences > min_bin)
        n_examples = mask.sum().item()
        if n_examples > 0:
            current_conf, current_pred = confidences[mask], predictions[mask]
            avg_conf = current_conf.mean()
            avg_acc = current_pred.mean()
            ece[min_bin] = (avg_acc.cpu().numpy(), avg_conf.cpu().numpy(), n_examples)
        else:
            ece[min_bin] = (0, 0, 0)
        # update bin
        max_bin = min_bin

    return sum([(float(nex) / float(confidences.shape[0])) * np.abs(acc - conf) for acc, conf, nex in ece.values()])
