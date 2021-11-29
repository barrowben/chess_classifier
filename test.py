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


### [NOT USED] ###
def divergence(class1, class2):
  """compute a vector of 1-D divergences
  
  class1 - data matrix for class 1, each row is a sample
  class2 - data matrix for class 2
  
  returns: d12 - a vector of 1-D divergence scores
  """

  # Compute the mean and variance of each feature vector element
  m1 = np.mean(class1, axis=0)
  m2 = np.mean(class2, axis=0)
  v1 = np.var(class1, axis=0)
  v2 = np.var(class2, axis=0)

  # Plug mean and variances into the formula for 1-D divergence.
  # (Note that / and * are being used to compute multiple 1-D
  #  divergences without the need for a loop)
  d12 = 0.5 * (v1 / v2 + v2 / v1 - 2) + 0.5 * (m1 - m2) * (m1 - m2) * (
    1.0 / v1 + 1.0 / v2
  )

  return d12

# def lle_reduce(X):
#   lle = LocallyLinearEmbedding(n_components=10, n_neighbors=10)
#   X_reduced = lle.fit_transform(X)
#   return X_reduced

# Remove knights if on edges (bad position in chess)
# MAKES CLASSIFICATION WORSE!
# if (square % 8 == 0) or ((square + 1) % 8 == 0):
#   nearest = nearest[(nearest != 'n') & (nearest != 'N')]

def main():
    """Train the classifier and save the model."""

    # Load the list of boards that will be used for training.
    with open("data/boards.train.json", "r", encoding="utf-8") as fp:
        board_metadata = json.load(fp)

    print("TESTING BEGINS")
    model_data = test(board_metadata, "data/clean")


if __name__ == "__main__":
    main()
