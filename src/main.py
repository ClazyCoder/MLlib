import argparse
import yaml
from src.trainers import build_trainer
import os
import logging
from src.utils.config import RootConfig

LOG_FILE_NAME = "mllib_logs.log"
LOG_DIR = os.environ.get("LOG_DIR", "./logs")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
LOG_FILE = os.path.join(LOG_DIR, LOG_FILE_NAME)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()])
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = RootConfig(**config)
    trainer = build_trainer(config)
    if args.mode == "train":
        logger.info(f"Training started.")
        trainer.train()
        logger.info(f"Training completed successfully.")
    elif args.mode == "test":
        logger.info(f"Testing started.")
        trainer.test()
        logger.info(f"Testing completed successfully.")


if __name__ == "__main__":
    main()
