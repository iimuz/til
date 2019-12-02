# third party
from scipy import io


def load_dataset() -> None:
    FILEPATH = "data/hs_bearing_1/hs_bearing_1/sensor-20130307T015746Z.mat"
    # var = h5py.File(FILEPATH, "r")
    var = io.loadmat(FILEPATH)
    for key, item in var.items():
        print(key, item)


if __name__ == "__main__":
    load_dataset()
