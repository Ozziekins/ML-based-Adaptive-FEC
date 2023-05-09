import numpy as np
from torch.utils.data import Dataset
import torch
class SequenceDataset(Dataset):
    def __init__(self, data, input_size, gap_size, output_size):
        gap_offset = gap_size + input_size
        subset_len = input_size+gap_size+output_size
        self.x, self.y = self._generate_slides(data, input_size, gap_offset, output_size, subset_len)
         
        
    def _generate_slides(self, data, input_size, gap_offset, output_size, subset_len):
        groupped = data.groupby(['client_user_id','session_id'])[['dropped_frames','FPS','bitrate','RTT', 'loss_rate']]
        # to use vstack we need to spoof the first elements
        # we will remove them later
        # -3 because gorup by removes 2 columns as well, as the last one is target
        x = np.array([np.zeros((input_size, data.shape[1]-2))])
        y = np.array([np.zeros((subset_len - gap_offset))])
        for k, v in groupped:
            numpy_data = v.to_numpy()
            # identifying seq length
            seq_length = numpy_data.shape[0]
            # shape - [n_seq, subset_len, num_features]
            observations = np.array([numpy_data[i*subset_len:(i+1)*subset_len, :] for i in range(seq_length // subset_len)])
            # Need to filter out cases where it is impossible to build windows
            if len(observations.shape) > 1:
                x = np.vstack([x, observations[:, :input_size, :]])
                y = np.vstack([y, observations[:, gap_offset:, -1]])
        return x[1:], y[1:]
            

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]).float(), torch.from_numpy(self.y[idx]).float()