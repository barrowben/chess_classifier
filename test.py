import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from utils import utils
import system
import evaluate
import train

def test(board_metadata: list, image_dir: str):
    images_train = utils.load_board_images(image_dir, board_metadata)
    labels_train = utils.load_board_labels(board_metadata)
    fvectors_train = system.images_to_feature_vectors(images_train)
    model_data = system.process_training_data(fvectors_train, labels_train)


def main():
    """Train the classifier and save the model."""

    # Load the list of boards that will be used for training.
    with open("data/boards.train.json", "r", encoding="utf-8") as fp:
        board_metadata = json.load(fp)

    print("TESTING BEGINS")
    model_data = test(board_metadata, "data/clean")


if __name__ == "__main__":
    main()
