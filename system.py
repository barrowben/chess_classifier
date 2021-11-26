"""Baseline classification system.

Solution outline for the COM2004/3004 assignment.

This solution will run but the dimensionality reduction and
the classifier are not doing anything useful, so it will
produce a very poor result.

version: v1.0
"""
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy.linalg import eig

N_DIMENSIONS = 10

def classify(train: np.ndarray, train_labels: np.ndarray, test: np.ndarray) -> List[str]:
    """Classify a set of feature vectors using a training set.

    This dummy implementation simply returns the empty square label ('.')
    for every input feature vector in the test data.

    Note, this produces a surprisingly high score because most squares are empty.

    Args:
        train (np.ndarray): 2-D array storing the training feature vectors.
        train_labels (np.ndarray): 1-D array storing the training labels.
        test (np.ndarray): 2-D array storing the test feature vectors.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """
    # Nearest neighbour classification
    label = nearest_neighbour(train, train_labels, test)

    return label


# The functions below must all be provided in your solution. Think of them
# as an API that it used by the train.py and evaluate.py programs.
# If you don't provide them, then the train.py and evaluate.py programs will not run.
#
# The contents of these functions are up to you but their signatures (i.e., their names,
# list of parameters and return types) must not be changed. The trivial implementations
# below are provided as examples and will produce a result, but the score will be low.


def reduce_dimensions(data: np.ndarray, model: dict) -> np.ndarray:
    """Reduce the dimensionality of a set of feature vectors down to N_DIMENSIONS.

    The feature vectors are stored in the rows of 2-D array data, (i.e., a data matrix).
    The dummy implementation below simply returns the first N_DIMENSIONS columns.

    Args:
        data (np.ndarray): The feature vectors to reduce.
        model (dict): A dictionary storing the model data that may be needed.

    Returns:
        np.ndarray: The reduced feature vectors.
    """
    # PCA assumes data is centred, so we need to subtract the mean
    data_centered = data - model["mean"]
    eigenvectors = np.array(model["eigenvectors"])

    # Project training set onto 10D plane
    W10 = eigenvectors.T[:,:10]
    reduced_data = data_centered.dot(W10)
    # print(reduced_data.shape)

    return reduced_data

def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """Process the labeled training data and return model parameters stored in a dictionary.

    Note, the contents of the dictionary are up to you, and it can contain any serializable
    data types stored under any keys. This dictionary will be passed to the classifier.

    Args:
        fvectors_train (np.ndarray): training data feature vectors stored as rows.
        labels_train (np.ndarray): the labels corresponding to the feature vectors.

    Returns:
        dict: a dictionary storing the model data.
    """

    # Check display of a chess tile
    # test_index = 7
    # square_image = np.reshape(fvectors_train[test_index, :], (50, 50), order="C") # Use C-Like index order
    # print (labels_train[test_index])
    # plt.matshow(square_image, cmap=cm.Greys_r)
    # plt.show()

    # Center the data
    mean = fvectors_train.mean(axis=0)
    data_centered = fvectors_train - mean
    
    # Get principal components - the rows of Vh are the eigenvectors
    U, S, Vh = np.linalg.svd(data_centered)
    e1 = Vh.T[:,0].tolist()
    e2 = Vh.T[:,1].tolist()
    e3 = Vh.T[:,2].tolist()
    e4 = Vh.T[:,3].tolist()
    e5 = Vh.T[:,4].tolist()
    e6 = Vh.T[:,5].tolist()
    e7 = Vh.T[:,6].tolist()
    e8 = Vh.T[:,7].tolist()
    e9 = Vh.T[:,8].tolist()
    e10 = Vh.T[:,9].tolist()

    # Create the model data
    model = {}
    model["labels_train"] = labels_train.tolist()
    model["mean"] = mean.tolist()
    model["eigenvectors"] = [e1, e2, e3, e4, e5, e6, e7, e8, e9, e10]
    fvectors_train_reduced = reduce_dimensions(fvectors_train, model)
    model["fvectors_train"] = fvectors_train_reduced.tolist()

    return model


def nearest_neighbour(chess_train_data, chess_train_labels, chess_test_data, features=None):
    if features is None:
        features = np.arange(0, chess_train_data.shape[1])

    chess_test_data = chess_test_data[:, features]
    chess_train_data = chess_train_data[:, features]
    
    # Nearest Neighbour
    x = np.dot(chess_test_data, chess_train_data.transpose())
    modtest = np.sqrt(np.sum(chess_test_data * chess_test_data, axis=1))
    modtrain = np.sqrt(np.sum(chess_train_data * chess_train_data, axis=1))
    dist = x / np.outer(modtest, modtrain.transpose())

    # cosine distance
    nearest = np.argmax(dist, axis=1)
    predicted_labels = chess_train_labels[nearest]

    return predicted_labels

# NOT USED - PCA WAS CHOSEN INSTEAD
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

def images_to_feature_vectors(images: List[np.ndarray]) -> np.ndarray:
    """Takes a list of images (of squares) and returns a 2-D feature vector array.

    In the feature vector array, each row corresponds to an image in the input list.

    Args:
        images (list[np.ndarray]): A list of input images to convert to feature vectors.

    Returns:
        np.ndarray: An 2-D array in which the rows represent feature vectors.
    """
    h, w = images[0].shape
    n_features = h * w
    fvectors = np.empty((len(images), n_features))
    for i, image in enumerate(images):
        fvectors[i, :] = image.reshape(1, n_features)

    return fvectors


def classify_squares(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in an arbitrary order.

    Note, the feature vectors stored in the rows of fvectors_test represent squares
    to be classified. The ordering of the feature vectors is arbitrary, i.e., no information
    about the position of the squares within the board is available.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    # Get some data out of the model. It's up to you what you've stored in here
    fvectors_train = np.array(model["fvectors_train"])
    labels_train = np.array(model["labels_train"])

    # Call the classify function.
    labels = classify(fvectors_train, labels_train, fvectors_test)

    return labels


def classify_boards(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in 'board order'.

    The feature vectors for each square are guaranteed to be in 'board order', i.e.
    you can infer the position on the board from the position of the feature vector
    in the feature vector array.

    In the dummy code below, we just re-use the simple classify_squares function,
    i.e. we ignore the ordering.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    return classify_squares(fvectors_test, model)
