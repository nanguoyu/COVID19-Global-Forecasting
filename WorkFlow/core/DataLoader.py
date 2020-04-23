"""
@File : DataLoader.py
@Author: Dong Wang
@Date : 2020/4/21
"""
from . import pd, np


class DataLoader(object):
    def __init__(self, filename=None, split=0.8):
        """
        DataLoader Constructor
        :param filename: the name of target CSV file
        :param split: split train and val
        """
        assert filename and filename.find('.csv') > 0, "CSV file is needed"

        data = pd.read_csv(filename)
        # TODO Consider flexible country selection
        data_China = data[data['Country_Region'] == 'China'].groupby(['Date']).agg(
            {'ConfirmedCases': ['sum'], 'DepartureFlight': ['mean']})
        day = data_China.shape[0]
        for i in range(day):
            if data_China.iloc[i].values[1] == 0:
                data_China.iloc[i] = [data_China.iloc[i].values[0],
                                      int((data_China.iloc[i - 1].values[1] + data_China.iloc[i + 1].values[1]) / 2)]
        # TODO Random splitting
        self._data = data_China.values
        self._train = self._data[:int(len(self._data) * split)]
        self._val = self._data[int(len(self._data) * split):]
        self._len_train = len(self._train)
        self._len_val = len(self._val)
        print("[DataLoader]: There are ", self._len_train, " examples for training and ", self._len_val, " for validation")

    def get_train_data(self, seq_len, normalise):
        """
         Create x, y train data windows
        :param seq_len: len of window
        :param normalise: boolean value
        :return: trainX, trainY
        """
        data_x = []
        data_y = []
        for i in range(self._len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def get_evaluate_data(self, seq_len, normalise):
        """
        Create x, y evaluate data windows
        """
        data_x = []
        data_y = []
        for i in range(self._len_val - seq_len):
            x, y = self._next_evaluate_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def _next_window(self, i, seq_len, normalise):
        """
        Generates the next data window from the given index location i
        :param i: ith window
        :param seq_len: len of window
        :param normalise: boolean value
        :return:
        """
        window = self._train[i:i + seq_len]
        window = self.normalise_windows(window, single_window=True)[0] if normalise else window
        x = window[:-1]
        y = window[-1, [0]]
        # y = window[-1]
        return x, y

    def _next_evaluate_window(self, i, seq_len, normalise):
        """Generates the next test data window from the given index location i"""
        window = self._val[i:i + seq_len]
        window = self.normalise_windows(window, single_window=True)[0] if normalise else window
        x = window[:-1]
        y = window[-1, [0]]
        # y = window[-1]
        return x, y

    def normalise_windows(self, window_data, single_window=False):
        """Normalise window with a base value of zero"""
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
                normalised_window.append(normalised_col)
            normalised_window = np.array(
                normalised_window).T  # reshape and transpose array back into original multidimensional format
            normalised_data.append(normalised_window)
        return np.array(normalised_data)
