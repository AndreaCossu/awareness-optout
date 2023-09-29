from torchvision.models import resnet18, vgg16_bn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Compose, Normalize
import torch
import argparse
import wandb
from utils import EarlyStopping, GamblersLoss, get_confidences, compute_confidences
from tqdm import tqdm


normalization = Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

# generate code to transform the parameters above in input argument with argparse library
parser = argparse.ArgumentParser()
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--eval_batch_size', type=int, default=128)
parser.add_argument('--train_metrics_every', type=int, default=100)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--lr', type=float, default=1e-3)

parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--tolerance', type=float, default=0.0)

parser.add_argument('--eval_confidence_threshold', type=float, default=0.1,
                    help='use during evaluation to filter out examples where the model is not confident enough')

parser.add_argument('--coverage', type=float, nargs='+',default=[100.,99.,98.,97.,95.,90.,85.,80.,75.,70.,60.,50.,40.,30.,20.,10.],
                    help='the expected coverages used to evaluated the accuracies after abstention')

parser.add_argument('--pretrain_epochs', type=int, default=0)

parser.add_argument('--method', type=str, choices=['none', 'gamblers'], default='none')
parser.add_argument('--gamblers_temperature', type=float, default=7,
                    help='temperature for the gamblers loss (o in the original paper). '
                         'This should be >1 and <= number of classes')

parser.add_argument('--use_test', action="store_true")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


run_name = f"{args.method}_eval{args.eval_confidence_threshold}"


@torch.no_grad()
def evaluate_with_threshold(mdl, method, loader, confidence_threshold=0.0):
    """
    Evaluate the model on examples where the model is at least confident as the threshold.
    If the model is not sufficiently confident on any example, a perfect loss and accuracy will be returned.
    :param mdl:
    :param loader:
    :param confidence_threshold: if 0.0, always evaluate.
    :return: accuracy, loss, number of predictions that surpassed confidence threshold
    """

    mdl.eval()
    correct = 0
    total = 0
    loss_sum = 0.

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = mdl(x)

        confidences = get_confidences(args.method, out)
        mask = confidences >= confidence_threshold
        num_predictions = mask.sum().item()
        if num_predictions == 0:
            continue

        if method == 'gamblers':
            out = out[:, :-1]

        loss = torch.nn.functional.cross_entropy(out[mask], y[mask]).item()
        correct += (out[mask].argmax(dim=-1) == y[mask]).sum().item()
        total += num_predictions
        loss_sum += loss

    if total == 0:
        return 1.0, 0.0, total
    else:
        acc = 100 * (correct / float(total))
        loss = loss_sum / float(total)
        return acc, loss, total


@torch.no_grad()
def evaluate_coverage(mdl, method, loader, coverages):
    mdl.eval()

    confidence_accs = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = mdl(x)

        confidences = get_confidences(args.method, out).cpu().numpy().tolist()

        if method == 'gamblers':
            out = out[:, :-1]

        correct = (out.argmax(dim=-1) == y).long().cpu().numpy().tolist()
        for conf, corr in zip(confidences, correct):
            confidence_accs.append((conf, corr))

    sorted_accs = sorted(confidence_accs, key=lambda el: el[0], reverse=True)

    coverages_accs = {}
    for c in coverages:
        num_elements = int(float(c/100) * len(sorted_accs))
        coveraged = sorted_accs[:num_elements]
        coverages_accs[c] = 100*(sum([el[1] for el in coveraged]) / num_elements)

    return coverages_accs


wandb.init(
    project="awareness",
    config=vars(args),
    name=run_name,
    # dir=save_dir,
)

num_classes = 10

model = resnet18(weights=None, num_classes=num_classes+int(args.method == 'gamblers')).to(device)
train_dataset = CIFAR10(root='/raid/a.cossu/datasets', train=True, download=True,
                        transform=Compose([ToTensor(), normalization]))
test_dataset = CIFAR10(root='/raid/a.cossu/datasets', train=False, download=True,
                       transform=Compose([ToTensor(), normalization]))

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

if args.method == 'gamblers':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


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
        model.train()
        out = model(x)

        if args.pretrain_epochs > epoch:
            loss = pretrain_criterion(out[:, :num_classes], y)
        else:
            loss = train_criterion(out, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += (out[:, :num_classes].argmax(dim=-1) == y).sum().item() / float(x.shape[0])
        if (i+1) % args.train_metrics_every == 0:
            train_acc /= float(args.train_metrics_every)
            train_loss /= float(args.train_metrics_every)
            wandb.log({"train/acc": train_acc, "train/loss": train_loss},
                      commit=i+args.train_metrics_every < len(train_loader))
            train_acc, train_loss = 0., 0.

    if args.method == 'gamblers' and epoch in [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275]:
        lr *= 0.5
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    wandb.log({"epoch": epoch}, commit=False)

    if args.pretrain_epochs <= epoch:
        valid_acc, valid_loss, num_predictions = evaluate_with_threshold(model, args.method, valid_loader, confidence_threshold=args.eval_confidence_threshold)
        wandb.log({f"valid{args.eval_confidence_threshold}/acc": valid_acc,
                   f"valid{args.eval_confidence_threshold}/loss": valid_loss,
                   f"valid{args.eval_confidence_threshold}/num_predictions": num_predictions,
                   f"valid{args.eval_confidence_threshold}/perc_predictions": 100* (num_predictions / float(len(valid_dataset)))}, commit=False)

        coverages = evaluate_coverage(model, args.method, valid_loader, args.coverage)
        coverages_table = wandb.Table(data=[[x, y] for x, y in coverages.items()], columns=["coverage", "acc"])
        wandb.log({f"valid_coverage/valid_coverage_{epoch}": wandb.plot.line(coverages_table, "coverage", "acc",
                                                              title=f"Valid Coverage epoch {epoch}")}, commit=False)

        confidences = compute_confidences(model, args.method, valid_loader, device)
        table = wandb.Table(data=[[s] for s in confidences.tolist()], columns=["confidence"])
        wandb.log({f"valid_confidence/confidence_{epoch}": wandb.plot.histogram(table, "confidence", title=f"Max valid confidence epoch {epoch}")},
                  commit=False)


    valid_acc, valid_loss, _ = evaluate_with_threshold(model, args.method, valid_loader)
    wandb.log({"valid_full/acc": valid_acc, "valid_full/loss": valid_loss})

    if stopping.update(valid_loss):
        print("Stopping training after epoch ", epoch+1)
        break

if args.use_test:
    test_acc, test_loss, _ = evaluate_with_threshold(model, args.method, test_loader)
    wandb.log({"test_full/acc": test_acc, "test_full/loss": test_loss}, commit=False)

    coverages = evaluate_coverage(model, args.method, test_loader, args.coverage)
    coverages_table = wandb.Table(data=[[x, y] for x, y in coverages.items()], columns=["coverage", "acc"])
    wandb.log({f"test_coverage/test_coverage": wandb.plot.line(coverages, "coverage", "acc",
                                                               title=f"Test Coverage")}, commit=False)

    test_acc, test_loss, num_predictions = evaluate_with_threshold(model, args.method, test_loader, confidence_threshold=args.eval_confidence_threshold)
    wandb.log({f"test_{args.eval_confidence_threshold}/acc": test_acc,
               f"test_{args.eval_confidence_threshold}/loss": test_loss,
               f"test_{args.eval_confidence_threshold}/num_predictions": num_predictions,
               f"test_{args.eval_confidence_threshold}/perc_predictions": 100 * (num_predictions / float(len(valid_dataset)))})

wandb.finish()
