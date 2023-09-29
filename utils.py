import torch


class EarlyStopping:
    def __init__(self, mode='min', patience=3, tolerance=0):
        self.stop_count = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.patience = patience
        self.tolerance = tolerance
        self.mode = mode
        self.stopped = False

    def update(self, current_value):
        if self.mode == 'min':
            no_improve = current_value >= self.best_value - self.tolerance
        else:
            no_improve = current_value <= self.best_value + self.tolerance
        if no_improve:
            self.stop_count += 1
        else:
            self.stop_count = 0
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
        :param temperature: >1, <= number of classes
        """
        self.temperature = temperature

    def __call__(self, out, target):
        probs = torch.softmax(out, dim=-1)
        outputs, reservation = probs[:, :-1], probs[:, -1]
        gain = torch.gather(outputs, dim=1, index=target.unsqueeze(1)).squeeze()
        doubling_rate = (gain + (reservation / self.temperature)).log()
        return -doubling_rate.mean()


@torch.no_grad()
def get_confidences(method, out):
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
def compute_confidences(mdl, method, loader, device):
    mdl.eval()
    confidences = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = mdl(x)
        cfd = get_confidences(method, out)
        confidences.append(cfd)

    confidences = torch.cat(confidences, dim=0).cpu().numpy()

    return confidences
