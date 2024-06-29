from utils import *

if __name__ == "__main__":

    # load configuration file
    config = load_config()

    # read data
    raw_dataset = read_raw_data(config)
    raw_dataset = raw_dataset[config['data_source']['columns']]

    # data defence
    check_data(raw_dataset, config)

    # data splitting 
    X = raw_dataset[config['data_source']['features']].copy()
    y = raw_dataset[config['data_source']['target_name']].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = config['data_source']['test_size'], 
                                                    random_state = config['data_source']['random_state'], stratify = y)

    # dump data
    dump_dataset(X_train, 'X_train.pkl', config=config)
    dump_dataset(y_train, 'y_train.pkl', config=config)
    dump_dataset(X_test, 'X_test.pkl', config=config)
    dump_dataset(y_test, 'y_test.pkl', config=config)

    print('data pipeline OK')