from utils import *

if __name__ == '__main__':

    # load configuration file
    config = load_config()

    # load dataset
    X_train = load_dataset('X_train.pkl', config=config)
    y_train = load_dataset('y_train.pkl', config=config)
    X_test = load_dataset('X_test.pkl', config=config)
    y_test = load_dataset('y_test.pkl', config=config)

    train_set, test_set = concat_dataset(X_train, y_train, X_test, y_test)

    # encoding features
    train_set_ohe = ohe_input(train_set, config['data_source']['cat_features'])
    test_set_ohe = ohe_input(test_set, config['data_source']['cat_features'])
    train_set = concat_table(train_set, train_set_ohe, config['data_source']['num_features'], config)
    test_set = concat_table(test_set, test_set_ohe, config['data_source']['num_features'], config)

    # sampling
    train_set_rus = rus_fit_resample(train_set)
    train_set_ros = ros_fit_resample(train_set)
    train_set_smote = smote_fit_resample(train_set)

    # dump file
    X_train = {
    'WithoutResampling' : train_set.drop(columns = config['data_source']['target_name']),
    'Undersampling' : train_set_rus.drop(columns = config['data_source']['target_name']),
    'Oversampling' : train_set_ros.drop(columns = config['data_source']['target_name']),
    'SMOTE' : train_set_smote.drop(columns = config['data_source']['target_name'])
    }

    y_train = {
        'WithoutResampling' : train_set[config['data_source']['target_name']],
        'Undersampling' : train_set_rus[config['data_source']['target_name']],
        'Oversampling' : train_set_ros[config['data_source']['target_name']],
        'SMOTE' : train_set_smote[config['data_source']['target_name']]
    }

    # dump train
    dump_dataset(X_train, 'X_train_feng.pkl', config = config)
    dump_dataset(y_train, 'y_train_feng.pkl', config = config)

    # dump test
    dump_dataset(test_set.drop(columns = config['data_source']['target_name']), 'X_test_feng.pkl', config = config)
    dump_dataset(test_set[config['data_source']['target_name']], 'y_test_feng.pkl', config = config)

    print('preprocessing OK')




