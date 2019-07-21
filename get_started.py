import xgboost as xgb

from logging import getLogger

logger = getLogger(__name__)


def main() -> None:
    data_train = xgb.DMatrix("_data/agaricus.txt.train")
    data_test = xgb.DMatrix("_data/agaricus.txt.test")

    param = {"max_depth": 2, "eta": 1, "objective": "binary:logistic"}
    num_round = 2

    model = xgb.train(param, data_train, num_round)
    predicts = model.predict(data_test)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    main()
