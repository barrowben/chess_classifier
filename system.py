"""Chess classification system.

+ Classifier: K Nearest Neighbour (Weighted)
+ Dimensionality Reduction: PCA

Note that both nearest neighbour using cosine distance and
k-nearest-neighbour using euclidean distance have been implemented.
To switch between the two, comment/uncomment the indicated lines in
the classifier function.

"""
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import mode

N_DIMENSIONS = 10
K_NEAREST = 6

def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
  """Process the labeled training data and return model parameters stored in a dictionary.

  Args:
      fvectors_train (np.ndarray): training data feature vectors stored as rows.
      labels_train (np.ndarray): the labels corresponding to the feature vectors.

  Returns:
      dict: a dictionary storing the model data.
  """
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

def reduce_dimensions(data: np.ndarray, model: dict) -> np.ndarray:
  """Reduce the dimensionality of a set of feature vectors down to N_DIMENSIONS.

  Args:
      data (np.ndarray): The feature vectors to reduce.
      model (dict): A dictionary storing the model data that may be needed.

  Returns:
      np.ndarray: The reduced feature vectors.
  """
  # PCA assumes data is centered, so we need to subtract the mean
  data_centered = data - model["mean"]
  eigenvectors = np.array(model["eigenvectors"])

  # Project training set onto 10D plane
  W10 = eigenvectors.T[:,:10]
  reduced_data = data_centered.dot(W10)

  # Produces poor results (Clean: 18.6%, Noisy: 42.6%)
  # reduced_data = lle_reduce(data)

  return reduced_data

def classify(train: np.ndarray, train_labels: np.ndarray, test: np.ndarray) -> List[str]:
  """Classify a set of feature vectors using a training set.

  Args:
      train (np.ndarray): 2-D array storing the training feature vectors.
      train_labels (np.ndarray): 1-D array storing the training labels.
      test (np.ndarray): 2-D array storing the test feature vectors.

  Returns:
      list[str]: A list of one-character strings representing the labels for each square.
  """
  # # Nearest Neighbour
  # label = nearest_neighbour(train, train_labels, test)

  # KNN - standard
  # label = k_nearest_neighbour(train, train_labels, test, K_NEAREST)

  # K Nearest Neighbour - Weighted
  label = k_nearest_neighbour_weighted(train, train_labels, test, K_NEAREST)

  return label

def nearest_neighbour(chess_train_data, chess_train_labels, chess_test_data):    
  # Nearest Neighbour
  x = np.dot(chess_test_data, chess_train_data.transpose())
  modtest = np.sqrt(np.sum(chess_test_data * chess_test_data, axis=1))
  modtrain = np.sqrt(np.sum(chess_train_data * chess_train_data, axis=1))
  dist = x / np.outer(modtest, modtrain.transpose())

  # cosine distance
  nearest = np.argmax(dist, axis=1)
  predicted_labels = chess_train_labels[nearest]

  return predicted_labels

def k_nearest_neighbour(data, labels, test, k):
  label = []
  for sample in test:
    difference = (data - sample)
    distances = np.sum(difference * difference, axis=1) # Leave out the sqrt as it's monotonic
    nearest = labels[np.argsort(distances)[:k]]
    label += mode(nearest)[0][0] # Modal neighbour

  return label

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

  Args:
    fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
    model (dict): A dictionary storing the model data.

  Returns:
    list[str]: A list of one-character strings representing the labels for each square.
  """
  fvectors_train = np.array(model["fvectors_train"])
  labels_train = np.array(model["labels_train"])
  number_boards = int(fvectors_test.shape[0] / 64) # This assumes only whole boards are passed in

  # Call the classify function.
  # labels = k_nearest_neighbour_weighted(fvectors_train, labels_train, fvectors_test, K_NEAREST)
  # labels = k_nearest_filtered(fvectors_train, labels_train, fvectors_test, number_boards, K_NEAREST)
  labels = k_nearest_neighbour_weighted_filtered(fvectors_train, labels_train, fvectors_test, number_boards, K_NEAREST)
  return labels

def k_nearest_filtered(data, labels, test, number_boards, k):
  """K Nearest Neighbour augmented with experiments employing domain-specific knowledge.
  The function filters out impossible and improbable classifications.
  Possibly overfitting to the training set."""
  label = []
  i = 0
  j = 64
  for board in range(number_boards):
    has_b_king = False
    has_b_queen = False
    b_rook_count = 0

    for square in range(i, j):
      difference = (data - test[square])
      distances = np.sum(difference * difference, axis=1) # Leave out the sqrt as it's monotonic
      nearest = labels[np.argsort(distances)]

      # Remove the pawns if they are in the top or bottom rows (illegal position)
      if (i <= square < (i + 8)) or ((j - 8) <= square < j):
        nearest = nearest[(nearest != 'p') & (nearest != 'P')]

      # If black king/queen was already identified, remove from candidates
      # k/q tends to be in top half of board
      if has_b_king:
        nearest = nearest[(nearest != 'k')]
      if has_b_queen:
        nearest = nearest[(nearest != 'q')]

      # If in top half, disregard White K
      # White K tends to be in bottom half
      if (i <= square < (i + 32)):
        nearest = nearest[(nearest != 'K')]

      # If two black rooks already classified, ignore
      if b_rook_count == 2:
        nearest = nearest[(nearest != 'r')]
        
      # If the nearest neighbours of second row contain black pawn, replace last neighbour with black pawn
      if ((i+8) <= square < (i+16)) and (np.count_nonzero(nearest[:k] == 'p')>=2):
        nearest[:k] = ['p']

      # If the nearest neighbours of penultimate row contain white pawn, replace last neighbour with white pawn
      if ((i+48) <= square < (i+54)) and (np.count_nonzero(nearest[:k] == 'P')>=1):
        nearest[:k] = ['P']

      filtered = nearest[:k]
      prediction = mode(filtered)[0][0]

      if prediction == 'k':
        has_b_king = True
      if prediction == 'q':
        has_b_queen = True
      if prediction == 'r':
        b_rook_count +=1

      label += prediction # Modal neighbour
    i = j
    j = j + 64
    
  return label

def k_nearest_neighbour_weighted(data, labels, test, k):
  """Nearest neighbour using weights based on inverse distances."""
  predicted_labels = []
  for sample in test:
    # Calculate distances and weights
    difference = (data - sample)
    distances = np.sum(difference * difference, axis=1) # Leave out the sqrt as it's monotonic
    distances_sorted_indexes = np.argsort(distances)
    nearest_neighbours = distances[distances_sorted_indexes][:k]
    nearest_labels = labels[distances_sorted_indexes][:k]
    inverse_distance = 1 / (nearest_neighbours + 0.0000000000001) # Add a small constant to avoid `div` 0
    weighted_distances = inverse_distance / np.sum(inverse_distance)

    # Sort label indexes
    label_indexes_sorted = np.argsort(nearest_labels)

    # Group unique labels
    sorted_labels_array = nearest_labels[label_indexes_sorted]

    # Get the classes, their starting indexes and frequencies
    classes, index_start, count = np.unique(sorted_labels_array, return_counts=True, return_index=True)
    classes_list = classes.tolist()

    # Split up indexes
    class_index = np.split(label_indexes_sorted, index_start[1:])
    class_index_list = [element.tolist() for element in class_index]

    # Put the class and summed corresponding weights into dict
    votes = {}
    for i in range(len(classes_list)):
      indexes = class_index_list[i]
      weights = map(weighted_distances.__getitem__, indexes)
      votes[classes_list[i]] = sum(list(weights))

    # Get the highest score and add label to predictions
    prediction = max(votes, key=votes.get)
    predicted_labels += prediction # Modal neighbour

  return predicted_labels

def k_nearest_neighbour_weighted_filtered(data, labels, test, number_boards, k):
  """K nearest neighbour with distance-based weights AND some filtering."""
  label = []
  i = 0
  j = 64

  # Get all classes and their frequencies
  frequencies = count_piece_frequency(labels)
  # print(frequencies)


  for board in range(number_boards):
    for square in range(i, j):
      difference = (data - test[square])
      distances = np.sum(difference * difference, axis=1) # Leave out the sqrt as it's monotonic
      distances_sorted_idx = np.argsort(distances)
      nearest_neighbours = distances[distances_sorted_idx]
      nearest_labels = labels[distances_sorted_idx]

      # This is a rather convoluted way of removing the pawns from the candidate list if they are illegal positions
      # In some cases it leads to +0.1% in the noisy data but it seems to not really do much
      if (i <= square < (i + 8)) or ((j - 8) <= square < j):
        classes, idx_start, count = np.unique(nearest_labels, return_counts=True, return_index=True)

        # Get the indexes of where 'P' and 'p' values start
        index_start_P = (idx_start[4])
        index_end_P = index_start_P + count[4]
        index_start_p = (idx_start[10])
        index_end_p = index_start_p + count[10]

        # Create indices for slicing, then slice and concatenate
        nn_init_P = nearest_neighbours[0:index_start_P]
        nn_tail_P = nearest_neighbours[index_end_P:]
        nearest_neighbours= np.concatenate((nn_init_P, nn_tail_P), axis=None)
        nn_init_p = nearest_neighbours[0:index_start_p]
        nn_tail_p = nearest_neighbours[index_end_p:]
        nearest_neighbours= np.concatenate((nn_init_p, nn_tail_p), axis=None)
        nn_init_P_lab = nearest_labels[0:index_start_P]
        nn_tail_P_lab = nearest_labels[index_end_P:]
        nearest_labels= np.concatenate((nn_init_P_lab, nn_tail_P_lab), axis=None)
        nn_init_p_lab = nearest_labels[0:index_start_p]
        nn_tail_p_lab = nearest_labels[index_end_p:]
        nearest_labels= np.concatenate((nn_init_p_lab, nn_tail_p_lab), axis=None)

      # Reduce to k nearest neighbours
      nearest_neighbours = nearest_neighbours[:k]
      nearest_labels = nearest_labels[:k]

      # print("###############################")
      # print("Distance to nearest neighbours:")
      # print(nearest_neighbours)

      # print("\nNearest Labels:")
      # print(nearest_labels)

      inverse_distance = 1 / (nearest_neighbours + 0.0000000000001) # Add a small constant to avoid `div` 0
      weighted_distances = inverse_distance / np.sum(inverse_distance)
      # print("\nWeighted Distances:")
      # print(weighted_distances)

      # creates an array of indices, sorted by unique element
      idx_sort = np.argsort(nearest_labels)

      # sorts labels array so all unique elements are together 
      sorted_labels_array = nearest_labels[idx_sort]

      # returns the unique values, the index of the first occurrence of a value, and the count for each element
      classes, idx_start, count = np.unique(sorted_labels_array, return_counts=True, return_index=True)
      classes_list = classes.tolist()

      # splits the indices into separate arrays
      class_idx = np.split(idx_sort, idx_start[1:])
      class_idx_list = [element.tolist() for element in class_idx]

      votes = {}
      for i in range(len(classes_list)):
        indexes = class_idx_list[i]
        weights = map(weighted_distances.__getitem__, indexes)
        votes[classes_list[i]] = sum(list(weights))

      nearest_one = max(votes, key=votes.get)
      # print("\nPrediction (Weighted):")
      # print(nearest_one)

      label += nearest_one # Modal neighbour

    i = j
    j = j + 64

  return label

def count_piece_frequency(labels: np.ndarray) -> np.ndarray:
  (piece, count) = np.unique(labels, return_counts=True)
  frequency = np.asarray((piece, count)).T
  d = dict(enumerate(frequency[:,1].astype(np.int).flatten(), 1))

  return frequency