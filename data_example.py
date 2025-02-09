from pathlib import Path
import sys
import random
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def load_train_data() -> tuple[Dataset, DataLoader]:
    training_data = datasets.FashionMNIST(
        root="data", train=True, download=True, transform=ToTensor()
    )
    batch_size = 64
    train_loader = DataLoader(training_data, batch_size=batch_size)

    return training_data, train_loader


def load_test_data() -> tuple[Dataset, DataLoader]:
    test_data = datasets.FashionMNIST(
        root="data", train=False, download=True, transform=ToTensor()
    )
    batch_size = 64
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return test_data, test_loader


def train(
    model: NeuralNetwork,
    train_loader: DataLoader,
    loss_fn: nn.CrossEntropyLoss,
    optimizer: torch.optim.SGD,
    device: str | torch.device,
):
    size = len(train_loader.dataset)  # type: ignore
    model.train()
    for batch, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(x)
            print(f"loss: {loss:>7f} {current:>5d}/{size:>5d}")


def test(
    model: NeuralNetwork,
    test_loader: DataLoader,
    loss_fn: nn.CrossEntropyLoss,
    device: str | torch.device,
):
    size = len(test_loader.dataset)  # type: ignore
    n_batches = len(test_loader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= n_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {100*correct:>0.1f}%, avg. loss: {test_loss:>8f}")


def run_training(
    model: NeuralNetwork,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    device: str | torch.device,
    save_path: Path,
):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 1.0e-3)

    for t in range(epochs):
        print(f"-------Epoch {t}-------")
        train(model, train_loader, loss_fn, optimizer, device)
        test(model, test_loader, loss_fn, device)
    print("Done!")
    print("Saving model...")
    torch.save(model.state_dict(), save_path)
    print(f"Saved to {save_path}")


def evaluate(
    model: NeuralNetwork,
    test_data: torch.utils.data.Dataset,
    device: str | torch.device,
):
    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    model.eval()
    index = int(len(test_data) * random.random())  # type: ignore
    print(f"Evaluating element at index: {index}")
    x, y = test_data[index][0], test_data[index][1]
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f"Predicted: {predicted} Actual: {actual}")


def main_train(model_path: Path):
    device = "cpu"
    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()

    print(f"Using device: {device}")
    print("Creating model...")
    model = NeuralNetwork().to(device)
    if model_path.exists():
        print("Loading model params...")
        model.load_state_dict(torch.load(model_path))
    print("Loading data...")
    train_data, train_loader = load_train_data()
    test_data, test_loader = load_test_data()
    print("Training...")
    run_training(model, train_loader, test_loader, 5, device, model_path)


def main_eval(model_path: Path):
    device = "cpu"
    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()

    print(f"Using device: {device}")
    print("Loading model...")
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    print("Loading test data...")
    test_data, _ = load_test_data()
    print("Evaluating...")
    evaluate(model, test_data, device)


if __name__ == "__main__":
    model_path = Path("model.sav")
    if len(sys.argv) != 2:
        print("This script needs one argument to run, either train or eval")
    elif sys.argv[1] == "train":
        main_train(model_path)
    elif sys.argv[1] == "eval":
        main_eval(model_path)
    else:
        print("The argument to this script should be either train or eval")
