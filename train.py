import argparse
import pathlib
import pickle
from typing import Dict, List

import data_loader
import torch
from discriminator import Discriminator
from generator import Generator
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm


def format_history(d_loss: float, g_loss: float):
    """ 1 回分のデータを整形して返す
    """
    return {"d_loss": d_loss, "g_loss": g_loss}


def generate(generator: torch.nn, z_dim: int, image_num: int, is_cuda: bool):
    """ generator から画像を生成する
    """
    z = set_device(torch.rand((image_num, z_dim)), is_cuda)
    with torch.no_grad():
        z = Variable(z)
        samples = generator(z).data.cpu()
    return samples


def parse_arguments() -> Dict[str, any]:
    """ 引数の設定を行います
    """
    parser = argparse.ArgumentParser(usage=f"Usage python {__file__}")
    parser.add_argument(
        "-b",
        "--batch_size",
        action="store",
        nargs=None,
        const=None,
        default=128,
        type=int,
        choices=None,
        dest="batch_size",
        help="set batch_size",
        metavar=None,
    )
    parser.add_argument(
        "-l",
        "--learning_rate",
        action="store",
        nargs=None,
        const=None,
        default=2e-4,
        type=float,
        choices=None,
        dest="learning_rate",
        help="set learning rate",
        metavar=None,
    )
    parser.add_argument(
        "-z",
        "--z_dim",
        action="store",
        nargs=None,
        const=None,
        default=62,
        type=int,
        choices=None,
        dest="z_dim",
        help="set dimension for input vector of generator.",
        metavar=None,
    )
    parser.add_argument(
        "-e",
        "--epochs",
        action="store",
        nargs=None,
        const=None,
        default=25,
        type=int,
        choices=None,
        dest="num_epoch",
        help="set number of training epoch.",
        metavar=None,
    )
    parser.add_argument(
        "-d",
        "--log_dir",
        action="store",
        nargs=None,
        const=None,
        default=pathlib.Path("logs"),
        type=pathlib.Path,
        choices=None,
        dest="log_dir",
        help="set direcotry path of logs.",
        metavar=None,
    )
    parser.add_argument(
        "-e",
        "--checkpoint_each",
        action="store",
        nargs=None,
        const=None,
        default=5,
        type=int,
        choices=None,
        dest="checkpoint_each",
        help="save checkpoint data each input value.",
        metavar=None,
    )
    parser.add_argument(
        "-i",
        "--checkpoint_images",
        action="store",
        nargs=None,
        const=None,
        default=64,
        type=int,
        choices=None,
        dest="checkpoint_images",
        help="set number of images created by generator each epoch.",
        metavar=None,
    )
    parser.add_argument(
        "-c",
        "--cuda",
        action="store_true",
        default=False,
        dest="is_cuda",
        help="if set this flag and cuda device is avaliable, use cuda.",
    )
    args = parser.parse_args()
    return args


def get_data_loader(batch_size: int) -> DataLoader:
    """ Get dataloader.

    Parameters
    ---
    batch_size : int
        バッチサイズ
    """
    # loader = data_loader.load_mnist(batch_size)
    loader = data_loader.load_icons(pathlib.Path("data/icons"), batch_size)

    return loader


def run():
    """ Run training.
    """
    args = parse_arguments()
    print(f"parameters: {args}")

    # create output folders
    args.log_dir.mkdir(exist_ok=True)

    # check cuda
    is_cuda = torch.cuda.is_available() and args.is_cuda
    print(f"cuda state: {is_cuda}")

    # initialize network
    generator = set_device(Generator(), is_cuda)
    discriminator = set_device(Discriminator(), is_cuda)

    # optimizer
    g_optimizer = optim.Adam(
        generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999)
    )
    d_optimizer = optim.Adam(
        discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999)
    )

    # loss
    criterion = nn.BCELoss()

    # datasets
    data_loader = get_data_loader(args.batch_size)

    # training
    history = []
    progress_bar = tqdm(range(args.num_epoch))
    for epoch in progress_bar:
        d_loss, g_loss = train_one_epoch(
            discriminator,
            generator,
            criterion,
            d_optimizer,
            g_optimizer,
            data_loader,
            args.batch_size,
            args.z_dim,
            is_cuda,
        )
        if epoch % args.checkpoint_each == 0:
            save_checkpoint(discriminator, generator, epoch, args.log_dir)
            save_image(
                generate(generator, args.z_dim, args.checkpoint_images, is_cuda),
                args.log_dir.joinpath(f"epoch_{epoch:03}.png"),
            )
        history.append(format_history(d_loss, g_loss))

        state_message = f"epoch {epoch}, d_loss: {d_loss:.4}, g_loss: {g_loss:.4}"
        progress_bar.set_description(state_message)

    save_history(history, args.log_dir)


def save_checkpoint(
    discriminator: nn.Module, generator: nn.Module, epoch: int, log_dir: pathlib.Path
) -> None:
    """ save checkpoint.
    """
    torch.save(
        generator.state_dict(), str(log_dir.joinpath(f"generator_{epoch:03}.pth"))
    )
    torch.save(
        discriminator.state_dict(),
        str(log_dir.joinpath(f"discriminator_{epoch:03}.pth")),
    )


def save_history(history: List[Dict[str, str]], log_dir: pathlib.Path) -> None:
    """ save history.
    """
    with open(log_dir.joinpath("history.pkl"), "wb") as f:
        pickle.dump(history, f)


def train_one_epoch(
    discriminator: nn.Module,
    generator: nn.Module,
    criterion,
    d_optimizer,
    g_optimizer,
    data_loader: DataLoader,
    batch_size: int,
    z_dim: int,
    is_cuda: bool,
):
    """ Learn 1 epoch.

    Parameters
    ---

    discriminator : nn.Module
    generator : nn.Module
    criterion
    d_optimizer
    g_optimizer
    data_loader
    batch_size : int
        バッチサイズ
    z_dim : int
        入力ランダムベクトルの次元数
    is_cuda : bool
        True の場合に、 cuda を利用する
    """
    # 訓練モード
    discriminator.train()
    generator.train()

    # ラベル設定
    label_real = set_device(Variable(torch.ones(batch_size, 1)), is_cuda)
    label_fake = set_device(Variable(torch.zeros(batch_size, 1)), is_cuda)

    data_size = 0
    d_running_loss = 0
    g_running_loss = 0
    for batch_idx, (real_images, _) in enumerate(data_loader):
        # 最後のバッチでバッチ数が足りない場合は無視する
        if real_images.size()[0] != batch_size:
            break
        data_size += batch_size

        # Discriminator が real 画像を real と判定できるか
        d_optimizer.zero_grad()
        d_real = discriminator(Variable(set_device(real_images, is_cuda)))
        d_real_loss = criterion(d_real, label_real)

        # Discriminator 用に fake 画像を生成し、
        # Discriminator が fake と判定できるか
        z = Variable(set_device(torch.rand((batch_size, z_dim)), is_cuda))
        fake_images = generator(z)
        d_fake = discriminator(fake_images.detach())
        d_fake_loss = criterion(d_fake, label_fake)

        # 2 つの loss の最小化
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        d_optimizer.step()
        d_running_loss += d_loss.data

        # Generator が生成した画像を Discriminator に real と判定させられるか
        g_optimizer.zero_grad()
        z = Variable(set_device(torch.rand((batch_size, z_dim)), is_cuda))
        fake_images = generator(z)
        d_fake = discriminator(fake_images)
        g_loss = criterion(d_fake, label_real)
        g_loss.backward()
        g_optimizer.step()
        g_running_loss += g_loss.data

    d_running_loss /= data_size
    g_running_loss /= data_size

    return d_running_loss, g_running_loss


def set_device(src, is_cuda):
    """ CUDA デバイス対応に変換する
    """
    if is_cuda:
        return src.cuda()
    return src


if __name__ == "__main__":
    run()
