from custom_preprocess import custom_preprocess
import torch
import os


def load_cached_or_preprocess_and_cache(df, path_to_folder='./cached_subsets/', file_names=['X_train_cached.pt', 'X_dev_cached.pt', 'X_test_cached.pt', 'y_train_cached.pt', 'y_dev_cached.pt', 'y_test_cached.pt']):

    if path_to_folder[-1] != '/':
        path_to_folder = path_to_folder + '/'



    # Check if there is need to cache subsets (preprocess and cache if need)
    were_cached = None
    for subset_file_name in file_names:
        if not os.path.exists(path_to_folder + subset_file_name):
            print(f'Not found file {subset_file_name} in given folder ({path_to_folder}), starting recaching...')
            subsets = custom_preprocess(
                df, dev_size=0.15, test_size=0.15, even_dist=True)

            print('Saving them (caching)...')
            X_train = subsets[0]
            X_dev = subsets[1]
            X_test = subsets[2]

            y_train = subsets[3]
            y_dev = subsets[4]
            y_test = subsets[5]

            if not os.path.exists(path_to_folder):
                os.makedirs(path_to_folder)

            for file_name, subset in zip(file_names, subsets):
                torch.save(subset, path_to_folder + file_name)
            print('Done! All subsets are preprocessed and cached')

            were_cached = True
            break

    # If all subsets are already cached - then just load them (that saves a lot of time)
    if not were_cached:
        print('All cached subsets were found, loading...')
        X_train = torch.load(path_to_folder + file_names[0])
        X_dev = torch.load(path_to_folder + file_names[1])
        X_test = torch.load(path_to_folder + file_names[2])

        y_train = torch.load(path_to_folder + file_names[3])
        y_dev = torch.load(path_to_folder + file_names[4])
        y_test = torch.load(path_to_folder + file_names[5])
        print('Loaded!')

    return X_train, X_dev, X_test, y_train, y_dev, y_test
