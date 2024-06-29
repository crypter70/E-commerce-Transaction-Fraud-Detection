import streamlit as st
import requests

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

            with st.spinner("Sending data to prediction server ..."):
                result = requests.post("http://localhost:8080/predict", json = raw_data).json()
                # result = requests.post("http://api:8080/predict", json = raw_data).json()
    
            if result["error_msg"] != "":
                st.error("Error Occurs While Predicting: {}".format(result["error_msg"]))
            else:
                if result["res"] != "Fraud":
                    st.success("Predicted: Not Fraud.")
                else:
                    st.error("Predicted: Fraud.")


if __name__ == "__main__":
    main()
