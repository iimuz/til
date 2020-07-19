"""学習用スクリプト."""
# default packages
import logging
import random
import traceback

# third party packages
import pytorch_lightning as pl
import pytorch_lightning.callbacks as pl_callbacks
import torch.cuda as torch_cuda
import torch.utils.data as torch_data
import torchvision.transforms as tv_transforms

# my packages
import src.data.mvtec_ad as mvtec_ad
import src.data.directories as directories
import src.data.log_utils as log_utils
import src.models.cnn_ae as cnn_ae
import src.models.vanila_vae as vanila_vae
import src.models.vanila_gan as vanila_gan
import src.models.trainer as trainer
import src.models.trainer_gan as trainer_gan

# logger
logger = logging.getLogger(__name__)


def worker_init_random(worker_id: int) -> None:
    random.seed(worker_id)


def main() -> None:
    """学習処理の実行スクリプト."""
    log_utils.init_root_logger()

    # train_dir = directories.get_raw().joinpath("hazelnut/train")
    # filelist = sorted(list(train_dir.glob("**/*.png")))
    train_dir = directories.get_raw().joinpath("img_align_celeba")
    filelist = sorted(list(train_dir.glob("**/*.jpg")))
    num_train = int(len(filelist) * 0.8)

    image_size = (64, 64)
    transforms = tv_transforms.Compose(
        [
            # tv_transforms.Grayscale(num_output_channels=1),
            tv_transforms.RandomHorizontalFlip(),
            tv_transforms.CenterCrop(148),
            tv_transforms.Resize(image_size),
            tv_transforms.ToTensor(),
            tv_transforms.Lambda(lambda x: 2.0 * x - 1.0),
        ]
    )
    dataset_train = mvtec_ad.Dataset(
        filelist[:num_train], transforms, mvtec_ad.Mode.TRAIN
    )
    dataset_valid = mvtec_ad.Dataset(
        filelist[num_train:], transforms, mvtec_ad.Mode.VALID
    )

    batch_size = 144
    num_workers = 4
    dataloader_train = torch_data.DataLoader(
        dataset_train,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_random,
    )
    dataloader_valid = torch_data.DataLoader(
        dataset_valid,
        batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_random,
    )

    random_seed = 42
    pl.seed_everything(random_seed)
    in_channels = 3
    out_channels = 3
    hparams = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "random_seed": random_seed,
    }
    # network = cnn_ae.SimpleCBR(in_channels, out_channels)
    network = vanila_vae.VAE(in_channels, out_channels, image_size)
    # generator = vanila_gan.Generator(62)
    # discriminator = vanila_gan.Discriminator(in_channels, 1)
    # model = trainer.AETrainer(network, hparams)
    model = trainer.VAETrainer(network, hparams)
    # model = trainer_gan.GANTrainer(generator, discriminator, hparams)
    model.set_dataloader(dataloader_train, dataloader_valid)

    log_dir = "vanila_gan"
    save_top_k = 5
    early_stop = True
    min_epochs = 30
    max_epochs = 10000
    progress_bar_refresh_rate = 1
    cache_dir = directories.get_interim().joinpath(log_dir)
    profiler = True  # if use detail profiler, pl_profiler.AdvancedProfiler()
    model_checkpoint = pl_callbacks.ModelCheckpoint(
        filepath=str(cache_dir),
        monitor="val_loss",
        save_top_k=save_top_k,
        save_weights_only=False,
        mode="min",
        period=1,
    )
    pl_trainer = pl.Trainer(
        early_stop_callback=early_stop,
        default_root_dir=str(cache_dir),
        fast_dev_run=False,
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        gpus=[0] if torch_cuda.is_available() else None,
        progress_bar_refresh_rate=progress_bar_refresh_rate,
        profiler=profiler,
        checkpoint_callback=model_checkpoint,
    )
    pl_trainer.fit(model)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())
