from engine_loader.dataset import EngineDataset
import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
# import sys

BATCH_SIZE: int = 64
TRAIN_H5_PATH: Path = Path("./proton_dataset.h5")
TRAIN_DF_PATH: Path = Path("./proton_dataset.parquet")
TEST_H5_PATH: Path = Path("./proton_test_dataset.h5")
TEST_DF_PATH: Path = Path("./proton_test_dataset.parquet")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Conv1d(1000, 500, 1),
            nn.Flatten(),
            nn.Linear(1500, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 6),
        )

    def forward(self, x: torch.Tensor):
        logits = self.stack(x)
        return logits


def load_data(cloud_path: Path, var_path: Path) -> tuple[EngineDataset, DataLoader]:
    engine_dset = EngineDataset(var_path, cloud_path)
    engine_loader = DataLoader(engine_dset, batch_size=BATCH_SIZE)
    return engine_dset, engine_loader


def train(
    model: NeuralNetwork,
    loader: DataLoader,
    loss_fn: nn.MSELoss,
    optimizer: torch.optim.Adam,
    device: str | torch.device,
):
    model.train()
    for batch, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test(
    model: NeuralNetwork,
    loader: DataLoader,
    loss_fn: nn.MSELoss,
    device: str | torch.device,
):
    size = len(loader.dataset)  # type: ignore
    n_batches = len(loader)
    model.eval()
    test_loss, test_error = 0, torch.zeros(6, device=device)

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            test_error += torch.mean((pred - y), dim=0)
    test_loss /= n_batches
    test_error /= size
    print(f"Test Error: Avg. Loss: {test_loss:>8f} Error per Parameter: {test_error}")  # type: ignore


def run_training(
    model: NeuralNetwork,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    device: str | torch.device,
    save_path: Path | None,
):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)

    for t in range(epochs):
        print(f"-------Epoch {t}-------")
        train(model, train_loader, loss_fn, optimizer, device)
        test(model, test_loader, loss_fn, device)
    print("Done!")
    if save_path is not None:
        print("Saving model...")
        torch.save(model.state_dict(), save_path)
    print(f"Saved to {save_path}")


def main_train(model_path: Path | None, load: bool):
    device = "cpu"
    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()

    print(f"Using device: {device}")
    print("Creating model...")
    model = NeuralNetwork().to(device)
    if model_path is not None and model_path.exists() and load:
        print("Loading model params...")
        model.load_state_dict(torch.load(model_path))
    print("Loading data...")
    _, train_loader = load_data(TRAIN_H5_PATH, TRAIN_DF_PATH)
    _, test_loader = load_data(TEST_H5_PATH, TEST_DF_PATH)
    print("Training...")
    run_training(model, train_loader, test_loader, 5, device, model_path)


if __name__ == "__main__":
    main_train(None, False)
