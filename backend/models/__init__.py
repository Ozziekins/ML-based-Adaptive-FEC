import torch
INP_SIZE = 24
GAP_SIZE = 2
PREDICT_SIZE = 7
SUB_LEN = INP_SIZE + GAP_SIZE + PREDICT_SIZE
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
NUM_CLUSTERS = 3
NUM_STEPS_KMEANS = 5