import pandas as pd
import yaml
import joblib

from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

CONFIG_DIR = "../config/config.yaml"

# BASIC FUNCTION
# load config
def load_config() -> dict: 
    try:
        with open(CONFIG_DIR, "r") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError as fe:
        raise RuntimeError("Parameters file not found in path.")
    return config

# pickle load
def pickle_load(file_path: str):
    return joblib.load(file_path)

# pickle dump
def pickle_dump(data, file_path: str) -> None:
    joblib.dump(data, file_path)

# load dataset
def load_dataset(file_name, config = load_config()):
    try:
        PATH = config['train_test_data']['directory'] + file_name
        file_load = joblib.load(PATH)
    except:
        PATH = '../' + config['train_test_data']['directory'] + file_name
        file_load = joblib.load(PATH)
    return file_load

# concat dataset
def concat_dataset(X_train, y_train, X_test, y_test, config = load_config()):
    train_set = pd.concat([X_train, y_train], axis = 1)
    test_set = pd.concat([X_test, y_test], axis = 1)
        
    return train_set, test_set 

# dump dataset
def dump_dataset(to_dump, file_name, config = load_config()):
    try:
        joblib.dump(to_dump, config['train_test_data']['directory'] + file_name)
    except:
        joblib.dump(to_dump, '../' + config['train_test_data']['directory'] + file_name)


# DATA PIPELINE
# read raw data
def read_raw_data(config = load_config()):
    raw_dataset = pd.DataFrame()

    raw_dataset_dir = '../' + config['data_source']['directory'] + config['data_source']['file_name']  
    raw_dataset = pd.read_csv(raw_dataset_dir, encoding='utf-8')
    
    return raw_dataset

# check data
def check_data(input_data, config = load_config()):
    # browser
    assert input_data['browser'][0] in config['data_defense']['browser']['value'] or\
        input_data['browser'][0] != '',\
        f"Browser must be in list {config['data_defense']['browser']['value']}, and cannot be empty."
    
    # source
    assert input_data['source'][0] in config['data_defense']['source']['value'] or\
        input_data['source'][0] != '',\
        f"Source must be in list {config['data_defense']['source']['value']}, and cannot be empty."
    
    # age
    assert input_data.age.between(config['data_defense']['age'][0], config['data_defense']['age'][1]).sum() == len(input_data),\
        "an error occurs in Age range."
    
    # sex
    assert input_data['sex'][0] in config['data_defense']['sex']['value'] or\
        input_data['sex'][0] != '',\
        f"Sex must be in list {config['data_defense']['sex']['value']}, and cannot be empty."
    
    # purchase_value
    assert input_data.purchase_value.between(config['data_defense']['purchase_value'][0], config['data_defense']['purchase_value'][1]).sum() == len(input_data),\
        "an error occurs in Purchase Value range."


# WebApp
# ohe for new data in WebApp
def ohe_input_new_data(X, config = load_config()):
    X_train_preprocess = load_dataset('X_train.pkl')

    ohe = OneHotEncoder(handle_unknown = 'ignore')
    ohe.fit(X_train_preprocess[config['data_source']['cat_features']])
    
    ohe_data_raw = ohe.transform(X[config['data_source']['cat_features']]).toarray()
    X_index = X.index
    X_features = ohe.get_feature_names_out()

    X_ohe = pd.DataFrame(ohe_data_raw, index = X_index, columns = X_features)

    X_num = X[config['data_source']['num_features']]
    X_combined = pd.concat([X_ohe, X_num], axis=1)
    
    return X_combined


# PREPROCESSING
# ohe
def ohe_input(set_data, cat_features):
    
    ohe = OneHotEncoder(handle_unknown = 'ignore')
    
    ohe.fit(set_data[cat_features])
    
    ohe_values = ohe.transform(set_data[cat_features]).toarray()
    
    set_data_index = set_data.index
    set_data_columns = ohe.get_feature_names_out()
    
    set_data_ohe = pd.DataFrame(ohe_values, index=set_data_index, columns=set_data_columns)
    
    return set_data_ohe

# concat table 
def concat_table(set_data, set_data_ohe, num_features, config = load_config()):
    set_data_concated = pd.concat([set_data_ohe, set_data[num_features]], axis=1)
    set_data_concated = pd.concat([set_data_concated, set_data[config['data_source']['target_name']]], axis=1)
    
    return set_data_concated

# RUS
def rus_fit_resample(set_data: pd.DataFrame, config = load_config()):
    set_data = set_data.copy()
    rus = RandomUnderSampler(random_state = config['data_source']['random_state'])

    x_rus, y_rus = rus.fit_resample(set_data.drop(config['data_source']['target_name'], axis = 1), 
                                    set_data[config['data_source']['target_name']])
    set_data_rus = pd.concat([x_rus, y_rus], axis = 1)
    return set_data_rus

# ROS
def ros_fit_resample(set_data: pd.DataFrame, config = load_config()):
    set_data = set_data.copy()
    ros = RandomOverSampler(random_state = config['data_source']['random_state'])

    x_ros, y_ros = ros.fit_resample(set_data.drop(config['data_source']['target_name'], axis = 1), 
                                    set_data[config['data_source']['target_name']])
    set_data_ros = pd.concat([x_ros, y_ros], axis = 1)
    return set_data_ros

# SMOTE
def smote_fit_resample(set_data: pd.DataFrame, config = load_config()):
    set_data = set_data.copy()
    sm = SMOTE(random_state = config['data_source']['random_state'])

    x_sm, y_sm = sm.fit_resample(set_data.drop(config['data_source']['target_name'], axis = 1),
                                 set_data[config['data_source']['target_name']])
    set_data_sm = pd.concat([x_sm, y_sm], axis = 1)
    return set_data_sm


# MODELING
# train model
def train_model(X_train, y_train, config = load_config()):
    param = config['final_model']['parameter']
    dt = DecisionTreeClassifier(**param)
    dt.fit(X_train, y_train)
    return dt

# evaluation model
def evaluation_model(model, X_test, y_test):
    y_test_pred = model.predict(X_test)
    report = classification_report(y_true = y_test,
                                   y_pred = y_test_pred)
    print(report)

# dump model
def dump_model(to_dump, model_name, config = load_config()):
    try:
        joblib.dump(to_dump, config['final_model']['model_directory'] + model_name)
    except:
        joblib.dump(to_dump, '../' + config['final_model']['model_directory'] + model_name)


print('utils OK')