import numpy as np
import pickle
import sklearn.metrics as skmetrics
import xgboost as xgb

from logging import getLogger

logger = getLogger(__name__)


def main() -> None:
    # load data
    data_train = xgb.DMatrix("_data/agaricus.txt.train")
    data_test = xgb.DMatrix("_data/agaricus.txt.test")

    # train
    param = {"max_depth": 2, "eta": 1, "objective": "binary:logistic"}
    num_round = 2
    evals = [(data_train, "train"), (data_test, "eval")]

    evals_result = {}
    model = xgb.train(
        param, data_train, num_round, evals=evals, evals_result=evals_result
    )

    # predict and calculate score
    predicted_probability = model.predict(data_test)

    predicted_binary = np.where(predicted_probability > 0.5, 1, 0)
    accuracy = skmetrics.accuracy_score(data_test.get_label(), predicted_binary)
    logger.info(f"accuracy score: {accuracy}")

    # output results
    logger.debug(evals_result)
    with open("_data/get-started_results.pkl", "wb") as f:
        pickle.dump(
            {
                "accuracy": accuracy,
                "train error": evals_result["train"]["error"],
                "eval error": evals_result["eval"]["error"],
            },
            f,
        )


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    main()
