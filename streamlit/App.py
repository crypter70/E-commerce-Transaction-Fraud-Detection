import streamlit as st
import pandas as pd
import requests
import joblib
from sklearn.preprocessing import OneHotEncoder

model_path = '../streamlit/Decision_Tree_Classifier.pkl'

def pickle_load(file_path: str):
    return joblib.load(file_path)

def prediction(model, data):
    y_pred = model.predict(data)
    
    return y_pred

def preprocessing(data):

    X_train = joblib.load('X_train.pkl')

    ohe = OneHotEncoder(handle_unknown = 'ignore')
    ohe.fit(X_train[['browser', 'source', 'sex']])
    
    ohe_data_raw = ohe.transform(data[['browser', 'source', 'sex']]).toarray()
    data_index = data.index
    data_features = ohe.get_feature_names_out()

    data_ohe = pd.DataFrame(ohe_data_raw, index = data_index, columns = data_features)

    data_num = data[['age', 'purchase_value']]
    data_preprocessed = pd.concat([data_ohe, data_num], axis=1)
    
    return data_preprocessed

def main():

    st.title("E-commerce Transaction Fraud Detection WebApp")
    st.markdown("This app will classify whether an e-commerce transaction is fraudulent or not")

    with st.form(key = "transaction_data_form"):

        Age = st.number_input(
            label = "1. Age: ",
            min_value = 18,
            max_value = 76,
            help = "Value range from 18 to 76"
        )

        Sex = st.radio(
            label = "2. Gender: ",
            options = (
                "M",
                "F"
            )
        )

        Broswer = st.selectbox(
            label = "3. Browser: ",
            options = (
                "Chrome", 
                "IE", 
                "Safari", 
                "FireFox", 
                "Opera"
            )
        )

        Source = st.selectbox(
            label = "4. Source: ",
            options = (
                "SEO", 
                "Ads", 
                "Direct"
            )
        )

        PurchaseValue = st.number_input(
            label = "5. Purchase Value: ",
            min_value = 9,
            max_value = 154,
            help = "Value range from 9 to 154"
        )
    
        submitted = st.form_submit_button("Predict")

        if submitted:
            
            raw_data = {
                "age": Age,
                "sex": Sex,
                "browser": Broswer,
                "source": Source,
                "purchase_value": PurchaseValue,
            }

            model = pickle_load(model_path)

            raw_data = pd.DataFrame([raw_data])
            data_preprocessed = preprocessing(raw_data)

            with st.spinner("Classifying the transaction ..."):
            
                result = prediction(model, data_preprocessed)
            
            if result != 1:
                st.success("Not Fraud.")
            else:
                st.error("Fraud.")


if __name__ == "__main__":
    main()
