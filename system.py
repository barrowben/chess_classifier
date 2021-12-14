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
import scipy.stats as stats
import operator

N_DIMENSIONS = 10
K_NEAREST = 5

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

  # Get the frequencies of the pieces and scale them relative to the number of squares
  number_squares = fvectors_train.shape[0]
  frequencies = count_piece_frequency(labels_train)
  freq_empty = frequencies[0][1].astype(np.int) / number_squares
  freq_B = frequencies[1][1].astype(np.int) / number_squares
  freq_K = frequencies[2][1].astype(np.int) / number_squares
  freq_N = frequencies[3][1].astype(np.int) / number_squares
  freq_P = frequencies[4][1].astype(np.int) / number_squares
  freq_Q = frequencies[5][1].astype(np.int) / number_squares
  freq_R = frequencies[6][1].astype(np.int) / number_squares
  freq_b = frequencies[7][1].astype(np.int) / number_squares
  freq_k = frequencies[8][1].astype(np.int) / number_squares
  freq_n = frequencies[9][1].astype(np.int) / number_squares
  freq_p = frequencies[10][1].astype(np.int) / number_squares
  freq_q = frequencies[11][1].astype(np.int) / number_squares
  freq_r = frequencies[12][1].astype(np.int) / number_squares

  # Create the model data
  model = {}
  model["labels_train"] = labels_train.tolist()
  model["mean"] = mean.tolist()
  model["eigenvectors"] = [e1, e2, e3, e4, e5, e6, e7, e8, e9, e10]
  fvectors_train_reduced = reduce_dimensions(fvectors_train, model)
  model["fvectors_train"] = fvectors_train_reduced.tolist()
  model["frequencies_scaled"] = [freq_empty, freq_B, freq_K, freq_N, freq_P, freq_Q, freq_R, freq_b, freq_k, freq_n, freq_p, freq_q, freq_r]

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
  data_centered = (data - model["mean"])
  eigenvectors = np.array(model["eigenvectors"])

  # Project onto 10D plane
  P10 = eigenvectors.T[:,:10]
  reduced_data = data_centered.dot(P10)

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

  # NN
  # label = nearest_neighbour(train, train_labels, test)

  # KNN
  # label = k_nearest_neighbour(train, train_labels, test, K_NEAREST)

  # Weighted KNN
  label = k_nearest_neighbour_weighted(train, train_labels, test, K_NEAREST)

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
  fvectors_train = np.array(model["fvectors_train"])
  labels_train = np.array(model["labels_train"])
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
  number_boards = int(fvectors_test.shape[0] / 64)
  frequencies_scaled = model["frequencies_scaled"]

  # Call the classification function
  # labels = k_nearest_neighbour_weighted(fvectors_train, labels_train, fvectors_test, K_NEAREST)
  # labels = k_nearest_filtered(fvectors_train, labels_train, fvectors_test, number_boards, K_NEAREST)
  labels = k_nearest_neighbour_weighted_filtered(fvectors_train, labels_train, fvectors_test, number_boards, frequencies_scaled, K_NEAREST)

  return labels

def nearest_neighbour(fvectors_train: np.ndarray, labels_train: np.ndarray, fvectors_test: np.ndarray) -> List[str]:  
  """ Find the nearest neighbour, using cosine distance.
  
    Args:
    fvectors_train (np.ndarray): An array of trainig data in which feature vectors are stored as rows.
    labels_train (np.ndarray): 1-D array storing the training labels.
    fvectors_test (np.ndarray): 2-D array storing the test feature vectors.

  Returns:
    predicted_labels (list[str]): A list of one-character strings representing the labels for each square.
  """  
  x = np.dot(fvectors_test, fvectors_train.transpose())
  modtest = np.sqrt(np.sum(fvectors_test * fvectors_test, axis=1))
  modtrain = np.sqrt(np.sum(fvectors_train * fvectors_train, axis=1))
  dist = x / np.outer(modtest, modtrain.transpose()) # Cosine distance
  nearest = np.argmax(dist, axis=1)
  predicted_labels = labels_train[nearest]

  return predicted_labels

def k_nearest_neighbour(fvectors_train: np.ndarray, labels_train: np.ndarray, fvectors_test: np.ndarray, k) -> List[str]:
  """ Find the k nearest neighbours, using euclidean distance.

  Args:
  fvectors_train (np.ndarray): An array of trainig data in which feature vectors are stored as rows.
  chess_train_labels (np.ndarray): 1-D array storing the training labels.
  fvectors_test (np.ndarray): 2-D array storing the test feature vectors.
  k (int): number of nearest neighbours to select.

  Returns:
    predicted_labels (list[str]): A list of one-character strings representing the labels for each square.
  """
  label = []
  for sample in fvectors_test:
    difference = (fvectors_train - sample)
    distances = np.sum(difference * difference, axis=1) # Euclidean distance leaving out the sqrt as it's monotonic and we only care about the order
    nearest = labels_train[np.argsort(distances)[:k]]
    label += stats.mode(nearest)[0][0] # Modal neighbour

  return label

# Find the k-nearest neighbours, using euclidean distance
# Also filters out illegal and improbable classifications
def k_nearest_filtered(fvectors_train: np.ndarray, labels_train: np.ndarray, fvectors_test: np.ndarray, number_boards, k) -> List[str]:
  """K Nearest Neighbour augmented with experiments employing domain-specific knowledge.
  The function filters out impossible and improbable classifications.

  The function is not actually used, but has been left in for demonstration purposes.
  It is also the most accurate classifier.

  Args:
    fvectors_train (np.ndarray): An array of training data in which feature vectors are stored as rows.
    labels_train (np.ndarray): the labels corresponding to the feature vectors.
    fvectors_test (np.ndarray): An array of test data in which feature vectors are stored as rows.
    number_boards (int): totaly number of complete boards in the test data set
    frequencies_scaled (list[int]): freqeuncies of labels in training data set, scaled by total number
    k (int): number of nearest neighbours to select from

  Returns:
    label (list[str]): A list of one-character strings representing the labels for each square.
  """
  label = []
  i = 0
  j = 64
  for board in range(number_boards):
    has_b_king = False
    has_b_queen = False
    b_rook_count = 0

    for square in range(i, j):
      difference = (fvectors_train - fvectors_test[square])
      distances = np.sqrt(np.sum(difference * difference, axis=1))
      nearest = labels_train[np.argsort(distances)]

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
      prediction = stats.mode(filtered)[0][0]

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

# Find the k-nearest neighbours, using euclidean distance
# Weights the votes according to inverse distances
def k_nearest_neighbour_weighted(fvectors_train: np.ndarray, labels_train: np.ndarray, fvectors_test: np.ndarray, k) -> List[str]:
  """K nearest neighbour with inverse distance-based weights and some filtering.
  Uses weights based on inverse distances and filters out illeagal and improbable classifications
  
  Args:
    fvectors_train (np.ndarray): An array of training data in which feature vectors are stored as rows.
    labels_train (np.ndarray): the labels corresponding to the feature vectors.
    fvectors_test (np.ndarray): An array of test data in which feature vectors are stored as rows.
    k (int): number of nearest neighbours to select from

  Returns:
    label (list[str]): A list of one-character strings representing the labels for each square.
  """
  predicted_labels = []
  for sample in fvectors_test:
    # Calculate distances and weights
    difference = (fvectors_train - sample)
    distances = np.sqrt(np.sum(difference * difference, axis=1)) # Sqrt is important to give accurate distances for weights
    distances_sorted_indexes = np.argsort(distances)
    nearest_neighbours = distances[distances_sorted_indexes][:k]
    nearest_labels = labels_train[distances_sorted_indexes][:k]
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
    predicted_labels += prediction

  return predicted_labels

def k_nearest_neighbour_weighted_filtered(fvectors_train: np.ndarray, labels_train: np.ndarray, fvectors_test: np.ndarray, number_boards: int, frequencies_scaled: List[int], k: int) -> List[str]:
  """K nearest neighbour with inverse distance-based weights and some filtering.
  Uses weights based on inverse distances and filters out illeagal and improbable classifications
  
  Args:
    fvectors_train (np.ndarray): An array of training data in which feature vectors are stored as rows.
    labels_train (np.ndarray): the labels corresponding to the feature vectors.
    fvectors_test (np.ndarray): An array of test data in which feature vectors are stored as rows.
    number_boards (int): totaly number of complete boards in the test data set
    frequencies_scaled (list[int]): freqeuncies of labels in training data set, scaled by total number
    k (int): number of nearest neighbours to select from

  Returns:
    label (list[str]): A list of one-character strings representing the labels for each square.
  """
  label = []
  i = 0
  j = 64

  for board in range(number_boards):
    has_b_king = False
    has_b_queen = False
    b_rook_count = 0
    b_bishop_count = 0
    b_pawn_count = 0

    for square in range(i, j):
      # Calculate distances and weights
      difference = (fvectors_train - fvectors_test[square])
      distances = np.sqrt(np.sum(difference * difference, axis=1))
      distances_sorted_idx = np.argsort(distances)
      nearest_neighbours = distances[distances_sorted_idx]
      nearest_labels = labels_train[distances_sorted_idx]

      # Remove the pawns from the candidate list if they are in illegal positions
      if (i <= square < (i + 8)) or ((j - 8) <= square < j):
        classes, index_start, count = np.unique(nearest_labels, return_counts=True, return_index=True)

        # Get the indexes of where 'P' and 'p' values start
        index_start_P = (index_start[4])
        index_end_P = index_start_P + count[4]
        index_start_p = (index_start[10])
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

      # Reduce to K neighbours, calculate inverse distances and weights
      nearest_neighbours = nearest_neighbours[:k]
      nearest_labels = nearest_labels[:k]
      inverse_distance = 1 / (nearest_neighbours + 0.0000000000001) # Add a small constant to avoid `div` 0
      weighted_distances = inverse_distance / np.sum(inverse_distance)

      # Sort label indices & group them
      label_indexes_sorted = np.argsort(nearest_labels)
      sorted_labels_array = nearest_labels[label_indexes_sorted]

      # Get the classes, their starting indices and frequencies
      classes, index_start, count = np.unique(sorted_labels_array, return_counts=True, return_index=True)
      classes_list = classes.tolist()

      # Split up indices
      class_idx = np.split(label_indexes_sorted, index_start[1:])
      class_idx_list = [element.tolist() for element in class_idx]

      # Put the class and summed corresponding weights into a dict
      votes = {}
      for i in range(len(classes_list)):
        indexes = class_idx_list[i]
        weights = map(weighted_distances.__getitem__, indexes)
        votes[classes_list[i]] = sum(list(weights))

        # Remove black king/queen/rooks if already identified for this board and they are not the only potential candidates
        # We remove just the black pieces because they are at the top of the board and tend to be identified first
        if has_b_king and "k" in votes and len(votes) > 1:
          votes.pop("k")
        if has_b_queen and "q" in votes and len(votes) > 1:
          votes.pop("q")
        if b_rook_count == 2 and "r" in votes and len(votes) > 1:
          votes.pop("r")
        if b_bishop_count == 2 and "b" in votes and len(votes) > 1:
          votes.pop("b")
        if b_pawn_count == 8 and "p" in votes and len(votes) > 1:
          votes.pop("p")        

      # Sort the votes by values, get the number of candidates and the labels of the first 2 candidates
      sorted_votes = sorted(votes.items(), key=operator.itemgetter(1), reverse=True)
      number_candidates = len(sorted_votes)
      candidate_labels = [tuple[0] for tuple in sorted_votes][:2]

      # Look at the number of candidates and the difference between the highest votes
      # If the difference is low, then give preference to the more common piece
      # As the number of candidates increases, the confidence drops, so more options are opened up
      if number_candidates == 2:
        if abs(sorted_votes[0][1] - sorted_votes[1][1]) < 0.03:
          if "P" in candidate_labels:
            votes["P"] += frequencies_scaled[4]
          elif "p" in candidate_labels:
            votes["p"] += frequencies_scaled[10]
      elif number_candidates == 3 and (abs(sorted_votes[0][1] - sorted_votes[1][1])) < 0.1:
          if "P" in candidate_labels:
            votes["P"] += frequencies_scaled[4]
          elif "p" in candidate_labels:
            votes["p"] += frequencies_scaled[10]
          elif "R" in candidate_labels:
            votes["R"] += frequencies_scaled[6]
          elif "r" in candidate_labels:
            votes["r"] += frequencies_scaled[12]
      elif number_candidates == 4 and (abs(sorted_votes[0][1] - sorted_votes[1][1])) < 0.3:
          if "." in candidate_labels:
            votes["."] += frequencies_scaled[0]
          elif "P" in candidate_labels:
            votes["P"] += frequencies_scaled[4]
          elif "p" in candidate_labels:
            votes["p"] += frequencies_scaled[10]
          elif "R" in candidate_labels:
            votes["R"] += frequencies_scaled[6]
          elif "r" in candidate_labels:
            votes["r"] += frequencies_scaled[12]
          elif "k" in candidate_labels:
            votes["k"] += frequencies_scaled[8]
          elif "B" in candidate_labels:
            votes["B"] += frequencies_scaled[1]
          elif "N" in candidate_labels:
            votes["N"] += frequencies_scaled[3]
          elif "b" in candidate_labels:
            votes["b"] += frequencies_scaled[7]
          elif "K" in candidate_labels:
            votes["K"] += frequencies_scaled[2]
          elif "n" in candidate_labels:
            votes["n"] += frequencies_scaled[9]
          elif "Q" in candidate_labels:
            votes["Q"] += frequencies_scaled[5]
          elif "q" in candidate_labels:
            votes["q"] += frequencies_scaled[11]

      # Get the highest score and add label to predictions
      prediction = max(votes, key=votes.get)

      if prediction == 'k':
        has_b_king = True
      if prediction == 'q':
        has_b_queen = True
      if prediction == 'r':
        b_rook_count += 1
      if prediction == 'b':
        b_bishop_count += 1
      if prediction == 'p':
        b_pawn_count += 1

      label += prediction

    i = j
    j = j + 64

  return label

def count_piece_frequency(labels: np.ndarray) -> np.ndarray:
  "Helper function to get classes and their frequencies"
  (piece, count) = np.unique(labels, return_counts=True)
  frequency = np.asarray((piece, count)).T

  return frequency