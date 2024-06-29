from utils import *

if __name__ == '__main__':
    
    # load configuration file
    config = load_config()

    # load train and test
    X_train = load_dataset('X_train_feng.pkl')
    y_train = load_dataset('y_train_feng.pkl')

    X_test = load_dataset('X_test_feng.pkl')
    y_test = load_dataset('y_test_feng.pkl')

    X_train = X_train['Oversampling']
    y_train = y_train['Oversampling']

    final_model = train_model(X_train, y_train)
    evaluation_model(final_model, X_test, y_test)
    dump_model(final_model, config['final_model']['model_name'])

    print('modeling OK')

    