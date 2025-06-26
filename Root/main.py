import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from src.model.model import Net , RNN_Net
from src.utils.train import train
from src.utils.test import test
import matplotlib.pyplot as plt


def main():

    # ─── Hardcoded Args ───────────────────────────────
    batch_size = 64
    test_batch_size = 1000
    epochs = 50
    lr = 0.001
    gamma = 0.7
    use_accel = torch.cuda.is_available()
    log_interval = 10
    save_model = True

    # ─── Device Setup ────────────────────────────────
    device = torch.device("cuda" if use_accel else "cpu")
    torch.manual_seed(1)


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_kwargs = {'batch_size': batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': test_batch_size, 'shuffle': False}
    if use_accel:
        cuda_kwargs = {'num_workers': 1, 'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # ─── Model and Optimizer ─────────────────────────
    model = RNN_Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    # ─── Training Loop ───────────────────────────────

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, device, train_loader, optimizer, epoch)
        val_loss, val_acc = test(model, device, test_loader)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

    if save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    # Loss Plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy over Epochs')
    plt.legend()

    plt.tight_layout()

    # Save to file
    plt.savefig("mnist_training_metrics.png", dpi=300)
    plt.close()  # prevent it from displaying in headless environments






if __name__ == "__main__":
    main()