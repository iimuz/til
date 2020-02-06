from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def run():
    transform = transforms.Compose([transforms.ToTensor()])

    dataset = datasets.ImageFolder("data/icons", transform=transform)
    loader = DataLoader(dataset, shuffle=False, batch_size=10, num_workers=1)
    mean = 0.
    std = 0.
    nb_samples = 0
    for data, _ in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples
    mean /= nb_samples
    std /= nb_samples

    print(f"mean: {mean}")
    print(f"std: {std}")


if __name__ == "__main__":
    run()
