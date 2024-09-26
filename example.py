from src.predictor import vTrain
from src.config import vTrainConfig

import logging

import argparse

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def main(args):
    config = vTrainConfig.load_from_file(args.config)

    sim = vTrain(config)

    result, breakdown = sim()
    pred_iter_time = max(result.values())/1000/1000
    
    logger.info(f"predicted iteration time: {pred_iter_time:.3f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c, --config", type=str, dest="config")
    args = parser.parse_args()

    main(args)