import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import numpy as np

from backend.models import *


class Preprocessor:
    def __init__(self):
        pass

    def _scale(self, data):
        # df.set_index('timestamp', inplace=True)
        ct = ColumnTransformer(
            [("num_preprocess", StandardScaler(), ["dropped_frames", "FPS", "bitrate", "RTT"])])
        data[["dropped_frames", "FPS", "bitrate", "RTT"]] = ct.fit_transform(data)
        data["loss_rate"] = data['loss_rate'] / 100
        return data

    def _preprocess(self, data):
        # Dropping unnamed column
        data = data.drop(['Unnamed: 0'], axis=1)
        # Removing outliers. Getting quantiles
        Q1 = data['dropped_frames'].quantile(0.001)
        Q3 = data['dropped_frames'].quantile(0.999)
        IQR = Q3 - Q1

        # Compute the upper and lower bounds for the dropped_frames column
        upper_bound = Q3 * IQR
        lower_bound = Q1 * IQR

        # Find outliers in the dropped_frames column
        outliers = data[(data['dropped_frames'] < lower_bound) | (data['dropped_frames'] > upper_bound)]
        outliers_ids = outliers[['client_user_id', 'session_id']].drop_duplicates()

        # Drop all rows from data that have matching client_user_id and session_id
        data = data[~data[['client_user_id', 'session_id']].isin(outliers_ids.to_dict('list')).all(1)]

        data = self._scale(data)
        return data

    def _generate_slides(self, data, input_size, gap_offset, subset_len):
        groupped = data.groupby(['client_user_id', 'session_id'])[
            ['dropped_frames', 'FPS', 'bitrate', 'RTT', 'loss_rate']]
        # to use vstack we need to spoof the first elements
        # we will remove them later
        # -3 because gorup by removes 2 columns as well, as the last one is target
        x = np.array([np.zeros((input_size, data.shape[1] - 3))])
        y = np.array([np.zeros((subset_len - gap_offset))])
        for k, v in groupped:
            numpy_data = v.to_numpy()
            # identifying seq length
            seq_length = numpy_data.shape[0]
            # shape - [n_seq, subset_len, num_features]
            observations = np.array(
                [numpy_data[i * subset_len:(i + 1) * subset_len, :] for i in range(seq_length // subset_len)])
            # Need to filter out cases where it is impossible to build windows
            if len(observations.shape) > 1:
                x = np.vstack([x, observations[:, :input_size, :]])
                y = np.vstack([y, observations[:, gap_offset:, -1]])
        return x[1:], y[1:]

    def preprocess(self, data, input_size=INP_SIZE, gap_offset=INP_SIZE + GAP_SIZE, subset_len=SUB_LEN):
        data = self._preprocess(data)
        return self._generate_slides(data, input_size, gap_offset, subset_len)
