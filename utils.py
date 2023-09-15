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