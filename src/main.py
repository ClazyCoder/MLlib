import argparse
from src.utils.config import Config
from src.trainers import build_trainer
import os
import logging

CONFIG_PATH = "/app/configs/"
LOG_FILE_NAME = "mllib_logs.log"
LOG_DIR = "/app/logs/"
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
    parser.add_argument("--config", type=str, default="default.yaml")
    args = parser.parse_args()
    config = Config(os.path.join(CONFIG_PATH, args.config))
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
