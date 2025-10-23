import argparse
from src.utils.config import Config
from src.trainers import build_trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    config = Config(args.config)
    trainer = build_trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
