import numpy as np
from sklearn.preprocessing import MinMaxScaler

WINDOW = 5

def _load_data(file):
    dataset_original = np.load(file)
    return dataset_original

def _preprocessing(dataset_original):
    # Set type as float 32
    dataset_original = np.float32(dataset_original)

    # Add a channel dimension
    dataset_original = np.expand_dims(dataset_original, axis=-1)

    # Repleace obstacle values with NaN to ignore it in normalization
    dataset_original[dataset_original==-1] = np.nan

    # Normalize data between 0 and 1
    scaler = MinMaxScaler()
    dataset_original = scaler.fit_transform(
            dataset_original.reshape(-1, dataset_original.shape[-1])
        ).reshape(dataset_original.shape)
    
    # Set obstacle as -1
    dataset_original[np.isnan(dataset_original)] = -1

    return dataset_original

def _train_test(dataset_original):
    # Get indexes to optimize memory
    indexes = np.arange(dataset_original.shape[0])
    
    np.random.seed(42)
    np.random.shuffle(indexes)
    
    # Split training and validation dataset
    train_index = indexes[: int(0.9 * dataset_original.shape[0])]
    val_index = indexes[int(0.9 * dataset_original.shape[0]) :]
    train_dataset = dataset_original[train_index]
    val_dataset = dataset_original[val_index]

    # Divide sequences with window size
    train_dataset = train_dataset.reshape(-1, WINDOW, train_dataset.shape[2], train_dataset.shape[3], 1)
    val_dataset = val_dataset.reshape(-1, WINDOW, val_dataset.shape[2], val_dataset.shape[3], 1)

    # Separate input frames and output frame
    X_train, y_train = train_dataset[:,:-1,:,:,:], train_dataset[:,-1:,:,:,:]
    X_val, y_val = val_dataset[:,:-1,:,:,:], val_dataset[:,-1:,:,:,:]

    return X_train, y_train, X_val, y_val

def print_dataset_dimension(data):
    num_sequences = data.shape[0]
    lenght_sequence = data.shape[1]
    frame_width = data.shape[2]
    frame_height = data.shape[3]
    
    print(f"{num_sequences=} {lenght_sequence=} {frame_width=} {frame_height=}")

def print_dataset_statistics(data):
    data_max = np.nanmin(data)
    data_min = np.nanmax(data)
    data_mean = np.nanmean(data)
    data_median = np.nanmedian(data)
    data_variance = np.nanvar(data)
    
    statistics = f"{data_max=:.5f} {data_min=:.5f} {data_mean=:.5f} {data_median=:.5f} {data_variance=:.5f}"

    print(statistics)

def get_dataset(file):
    # TODO: Split this function
    dataset_original = _load_data(file)
    print("Original Dataset:")
    print_dataset_dimension(dataset_original)
    print_dataset_statistics(dataset_original)

    dataset_processed = _preprocessing(dataset_original)
    print("Processed Dataset:")
    print_dataset_dimension(dataset_original)
    print_dataset_statistics(dataset_original)

    X_train, y_train, X_val, y_val = _train_test(dataset_processed)
    print("X_train:")
    print_dataset_dimension(X_train)
    print("y_train:")
    print_dataset_dimension(y_train)
    print("X_val:")
    print_dataset_dimension(X_val)
    print("y_val:")
    print_dataset_dimension(y_val)

    return dataset_processed, X_train, y_train, X_val, y_val
