from torchvision.datasets import CIFAR100
from torchvision.models import resnet101
from torchvision.transforms import ToTensor, Compose, Normalize, RandomCrop, RandomHorizontalFlip
import torch
import argparse
import wandb
from utils import (EarlyStopping, GamblersLoss, compute_dataset_confidences_predictions, evaluate_coverage,
                   evaluate_with_threshold, evaluate_calibration)
from tqdm import tqdm


# generate code to transform the parameters above in input argument with argparse library
parser = argparse.ArgumentParser()
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--eval_batch_size', type=int, default=128)
parser.add_argument('--metrics_epoch_frequency', type=int, default=1)
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=5e-4)

parser.add_argument('--patience', type=int, default=6)
parser.add_argument('--tolerance', type=float, default=0.05)

parser.add_argument('--eval_confidence_threshold', type=float, nargs='+',
                    default=[0.90,0.80,0.70,0.60,0.50,0.40,0.30,0.20,0.10, 0.0],
                    help='use during evaluation to filter out examples where the model is not confident enough')

parser.add_argument('--coverage', type=float, nargs='+',
                    default=[100.,99.,98.,97.,95.,90.,85.,80.,75.,70.,60.,50.,40.,30.,20.,10.],
                    help='the expected coverages used to evaluated the accuracies after abstention')

parser.add_argument('--pretrain_epochs', type=int, default=0)

parser.add_argument('--method', type=str, choices=['none', 'gamblers'], default='none')
parser.add_argument('--gamblers_temperature', type=float, default=7,
                    help='temperature for the gamblers loss (o in the original paper). '
                         'This should be >1 and <= number of classes')

parser.add_argument('--use_test', action="store_true")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


run_name = f"{args.method}"
if args.method == 'gamblers':
    run_name += f"_temp{args.gamblers_temperature}"

wandb.init(
    project="awareness",
    config=vars(args),
    name='cifar100_'+run_name,
    tags=['resnet101']
)

num_classes = 100

model = resnet101(weights=None, num_classes=num_classes+int(args.method == 'gamblers')).to(device)


transform_train = Compose([
    RandomCrop(32, padding=4),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

transform_test = Compose([
    ToTensor(),
    Normalize( (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

train_dataset = CIFAR100(root='/raid/a.cossu/datasets', train=True, download=True,
                        transform=transform_train)
test_dataset = CIFAR100(root='/raid/a.cossu/datasets', train=False, download=True,
                       transform=transform_test)

train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [0.7, 0.3])

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=args.train_batch_size,
                                           shuffle=True,
                                           drop_last=False)
valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                           batch_size=args.eval_batch_size,
                                           shuffle=False,
                                           drop_last=False)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=args.eval_batch_size,
                                          shuffle=False,
                                          drop_last=False)

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
lr = args.lr

if args.method == 'gamblers':
    train_criterion = GamblersLoss(args.gamblers_temperature)
elif args.method == 'none':
    train_criterion = torch.nn.CrossEntropyLoss()
else:
    raise ValueError("Wrong method name.")
pretrain_criterion = torch.nn.CrossEntropyLoss()

stopping = EarlyStopping(mode='min', patience=args.patience, tolerance=args.tolerance)

for epoch in range(args.epochs):
    train_loss, train_acc = 0., 0.
    model.train()
    for i, (x, y) in enumerate(tqdm(train_loader)):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)

        if args.pretrain_epochs > epoch:
            loss = pretrain_criterion(out[:, :num_classes], y)
        else:
            loss = train_criterion(out, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += (out[:, :num_classes].argmax(dim=-1) == y).sum().item() / float(x.shape[0])

    train_acc /= float(len(train_loader))
    train_loss /= float(len(train_loader))
    wandb.log({"train/acc": train_acc, "train/loss": train_loss}, commit=False)

    if args.pretrain_epochs <= epoch and (epoch+1) % args.metrics_epoch_frequency == 0:
        confidences, predictions, losses = compute_dataset_confidences_predictions(model, args.method, valid_loader, train_criterion, device)
        table = wandb.Table(data=[[s] for s in confidences.cpu().numpy().tolist()], columns=["confidence"])
        wandb.log({f"valid_confidence/confidence_{epoch}": wandb.plot.histogram(table, "confidence", title=f"Valid confidence epoch {epoch}")},
                  commit=False)
        for conf in args.eval_confidence_threshold:
            valid_acc, valid_loss, num_predictions = evaluate_with_threshold(confidences, predictions, losses, confidence_threshold=conf)
            wandb.log({f"valid{conf}/acc": valid_acc,
                       f"valid{conf}/loss": valid_loss,
                       f"valid{conf}/num_predictions": num_predictions,
                       f"valid{conf}/perc_predictions": 100 * (num_predictions / float(len(valid_dataset)))}, commit=False)
            if conf == 0.0:
                stopping.update(valid_loss)

        coverages = evaluate_coverage(confidences, predictions, args.coverage)
        coverages_table = wandb.Table(data=[[x, y] for x, y in coverages.items()], columns=["coverage", "acc"])
        wandb.log({f"valid_coverage": wandb.plot.line(coverages_table, "coverage", "acc", title=f"Valid Coverage")}, commit=False)

        calibration = evaluate_calibration(confidences, predictions, bins=args.eval_confidence_threshold)
        wandb.log({"valid_calibration": calibration}, commit=False)

    wandb.log({"epoch": epoch})

    if stopping.stopped:
        print("Stopping training after epoch ", epoch+1)
        break

if args.use_test:
    confidences, predictions, losses = compute_dataset_confidences_predictions(model, args.method, test_loader, train_criterion, device)
    table = wandb.Table(data=[[s] for s in confidences.cpu().numpy().tolist()], columns=["confidence"])
    wandb.log({f"test_confidence/confidence": wandb.plot.histogram(table, "confidence", title=f"Test confidence")}, commit=False)

    coverages = evaluate_coverage(confidences, predictions, args.coverage)
    coverages_table = wandb.Table(data=[[x, y] for x, y in coverages.items()], columns=["coverage", "acc"])
    wandb.log({f"test_coverage/test_coverage": wandb.plot.line(coverages, "coverage", "acc",
                                                               title=f"Test Coverage")}, commit=False)

    for conf in args.eval_confidence_threshold:
        test_acc, test_loss, num_predictions = evaluate_with_threshold(confidences, predictions, losses, confidence_threshold=conf)
        wandb.log({f"test_{conf}/acc": test_acc,
                   f"test{conf}/loss": valid_loss,
                   f"test_{conf}/num_predictions": num_predictions,
                   f"test_{conf}/perc_predictions": 100 * (num_predictions / float(len(valid_dataset)))},
                  commit=False)

    calibration = evaluate_calibration(confidences, predictions, bins=args.eval_confidence_threshold)
    wandb.log({"test_calibration": calibration})


wandb.finish()
