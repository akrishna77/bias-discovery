from __future__ import print_function
from pathlib import Path
import argparse
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import wandb

from torchvision import transforms

from datasets.CelebADataset import CelebADataset
from models.celeba_classifier import CelebAClassifier

wandb.init(project="bias_discovery")
run_name = wandb.run.name

PROJ_ROOT = Path(__file__).resolve().parent.parent.parent

# Training settings
parser = argparse.ArgumentParser(description="Bias Discovery")
# Hyperparameters
parser.add_argument("--lr", type=float, metavar="LR", help="learning rate")
parser.add_argument("--momentum", type=float, metavar="M", help="SGD momentum")
parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay hyperparameter")
parser.add_argument("--batch_size", type=int, metavar="N", help="input batch size for training")
parser.add_argument("--epochs", type=int, metavar="N", help="number of epochs to train")
parser.add_argument("--hidden-dim", type=int, help="number of hidden features/activations")
parser.add_argument("--drop_rate", type=float, metavar="DR", help="data drop rate")
parser.add_argument("--save_dir", help="directory to save model in")
parser.add_argument("--kernel-size", type=int, help="size of convolution kernels/filters")
# Other configuration
parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
parser.add_argument(
    "--test-batch-size", type=int, default=1000, metavar="N", help="input batch size for testing (default: 1000)"
)
parser.add_argument(
    "--log-interval", type=int, default=10, metavar="N", help="number of batches between logging train status"
)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

dataroot = f"{PROJ_ROOT}/celeba/img_align_celeba/"
workers = 4
batch_size = args.batch_size
image_size = 224

dataset = CelebADataset(
    csv_file=f"{PROJ_ROOT}/celeba/list_attr_celeba.txt",
    root_dir=dataroot,
    transform=transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]
    ),
)

indices = list(range(len(dataset)))
train_indices, val_indices, test_indices = indices[:162770], indices[162771:182637], indices[182638:]

train_dataset, val_dataset, test_dataset = (
    torch.utils.data.Subset(dataset, train_indices),
    torch.utils.data.Subset(dataset, val_indices),
    torch.utils.data.Subset(dataset, test_indices),
)

biased_df_source_file = f"{PROJ_ROOT}/celeba/biased_celeba_black_hair/biased_celeba_black_hair.csv"

if os.path.isfile(biased_df_source_file) == False:
    if not os.path.exists(f"{PROJ_ROOT}/celeba/biased_celeba_black_hair/"):
        os.makedirs(f"{PROJ_ROOT}/celeba/biased_celeba_black_hair/")

    print(f"Training data size before biasing: {len(train_dataset)}")
    df = pd.read_csv(f"{PROJ_ROOT}/celeba/list_attr_celeba.txt", delimiter="\s+", header=0)
    df.replace(to_replace=-1, value=0, inplace=True)

    train_set_df = df.copy()
    train_set_df = train_set_df[:162770]

    train_set_df.drop(
        train_set_df[(train_set_df["Male"] == 0) & (train_set_df["Smiling"] == 1)].sample(frac=0.7).index,
        inplace=True,
    )
    train_set_df.drop(
        train_set_df[(train_set_df["Male"] == 1) & (train_set_df["Smiling"] == 0)].sample(frac=0.7).index,
        inplace=True,
    )

    train_set_df.to_csv(biased_df_source_file, index=False)

# train_dataset = CelebADataset(
#     csv_file=biased_df_source_file,
#     root_dir=dataroot,
#     transform=transforms.Compose(
#         [
#             transforms.Resize(image_size),
#             transforms.ToTensor(),
#         ]
#     ),
# )

print(f"Training data size after biasing: {len(train_dataset)}")

# DataLoaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

model = CelebAClassifier()

# cross-entropy loss function
criterion = F.cross_entropy
if args.cuda:
    model.cuda()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

wandb.watch(model)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.99, weight_decay=args.weight_decay)


def train(e):
    """
    Train the model for one epoch.
    """
    # Some models use slightly different forward passes and train and test
    # time (e.g., any model with Dropout). This puts the model in train mode
    # (as opposed to eval mode) so it knows which one to use.
    model.train()

    # train loop
    for batch_idx, batch in enumerate(train_loader):
        # prepare data
        images, targets = batch["image"], batch["attributes"]

        if args.cuda:
            images, targets = images.cuda(), targets.cuda()

        optimizer.zero_grad()
        loss = criterion(model(images), targets)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            val_loss, val_acc = evaluate("val", n_batches=4)
            train_loss = loss.data
            examples_this_epoch = batch_idx * len(images)
            epoch_progress = 100.0 * batch_idx / len(train_loader)
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\t"
                "Train Loss: {:.6f}\tVal Loss: {:.6f}\tVal Acc: {}".format(
                    e, examples_this_epoch, len(train_loader.dataset), epoch_progress, train_loss, val_loss, val_acc
                )
            )
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc})


@torch.no_grad()
def evaluate(split, verbose=False, n_batches=None):
    """
    Compute loss on val or test data.
    """
    model.eval()
    loss = 0
    correct = 0
    n_examples = 0
    if split == "val":
        loader = val_loader
    elif split == "test":
        loader = test_loader
    for batch_i, batch in enumerate(loader):
        data, target = batch["image"], batch["attributes"]
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        # data, target = Variable(data), Variable(target)
        output = model(data)
        loss += criterion(output, target, size_average=False).data
        # predict the argmax of the log-probabilities
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        # incorrect += pred.ne(target.data.view_as(pred)).cpu().sum()
        # incorrect_idx = np.argwhere(np.asarray(pred.ne(target.data.view_as(pred)).cpu().flatten()))
        # incorrect_list.append([names[i[0]] for i in incorrect_idx])
        n_examples += pred.size(0)
        if n_batches and (batch_i >= n_batches):
            break

    # incorrect_list = list(itertools.chain.from_iterable(incorrect_list))
    loss /= n_examples
    acc = 100.0 * correct / n_examples
    if verbose:
        print(
            "\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(split, loss, correct, n_examples, acc)
        )
    return loss, acc


# train the model one epoch at a time
for epoch in range(1, args.epochs + 1):
    train(epoch)

    torch.save(model.state_dict(), f"{args.save_dir}{run_name}-{epoch}.pth")

    evaluate("test", verbose=True)
