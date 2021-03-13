"""データセット取得のためのモジュール."""
# third party packages
import torchvision.datasets as tds
import torchvision.transforms as tvt


def get_dataset():
    """データセットを取得する."""
    MEAN = [0.49139968, 0.48215827, 0.44653124]
    STD = [0.24703233, 0.24348505, 0.26158768]

    transforms = [
        tvt.RandomCrop(32, padding=4),
        tvt.RandomHorizontalFlip(),
    ]
    normalize = [
        tvt.ToTensor(),
        tvt.Normalize(MEAN, STD),
    ]

    train_transform = tvt.Compose(transforms + normalize)
    valid_transform = tvt.Compose(normalize)

    dataset_train = tds.CIFAR10(
        root="data", train=True, download=True, transform=train_transform
    )
    dataset_valid = tds.CIFAR10(
        root="data", train=False, download=True, transform=valid_transform
    )

    return dataset_train, dataset_valid
