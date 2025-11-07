import argparse
from src.utils.config import Config
from src.trainers import build_trainer
import os

CONFIG_PATH = "/app/configs/"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--config", type=str, default="default.yaml")
    args = parser.parse_args()
    config = Config(os.path.join(CONFIG_PATH, args.config))
    trainer = build_trainer(config)
    if args.mode == "train":
        trainer.train()
    elif args.mode == "test":
        trainer.test()


if __name__ == "__main__":
    main()
