import json
import os
import glob
import argparse

IMAGE_EXTENSIONS = ['.jpg', '.png', '.jpeg', '.webp', '.gif', '.bmp', '.tiff']


def build_image_dataset(data_path, output_path):
    image_paths = [f for f in glob.glob(os.path.join(
        data_path, '**/*.*'), recursive=True) if os.path.isfile(f) and os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {data_path}")
    # TODO : find better way to get labels(NOW: using upper folder name as label)
    labels = [os.path.basename(os.path.dirname(image_path)) for image_path in image_paths]
    labels = list(map(int, labels))
    with open(output_path, 'w') as f:
        json.dump({'image_paths': image_paths, 'labels': labels}, f, indent=4)


def build_text_dataset(data_path, output_path):
    text_paths = [f for f in glob.glob(os.path.join(
        data_path, '**/*.txt'), recursive=True) if os.path.isfile(f)]
    if len(text_paths) == 0:
        raise ValueError(f"No text files found in {data_path}")
    # TODO : find better way to get labels(NOW: using upper folder name as label)
    labels = [os.path.basename(os.path.dirname(text_path)) for text_path in text_paths]
    labels = list(map(int, labels))
    with open(output_path, 'w') as f:
        json.dump({'text_paths': text_paths, 'labels': labels}, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="image")
    parser.add_argument("--data_path", type=str, default="data/images")
    parser.add_argument("--output_path", type=str, default="data/images.json")
    args = parser.parse_args()
    if args.mode == "image":
        build_image_dataset(args.data_path, args.output_path)
    elif args.mode == "text":
        build_text_dataset(args.data_path, args.output_path)
