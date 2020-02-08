import torch


def _main() -> None:
    print("current device: {}".format(torch.cuda.current_device()))
    print("device: {}".format(torch.cuda.device(0)))
    print("device count: {}".format(torch.cuda.device_count()))
    print("device name: {}".format(torch.cuda.get_device_name()))
    print("available: {}".format(torch.cuda.is_available()))


if __name__ == "__main__":
    _main()
