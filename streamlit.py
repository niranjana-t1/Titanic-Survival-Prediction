import streamlit as st
import pandas as pd
import pickle
from streamlit_lottie import st_lottie
import requests

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1518837695005-2083093ee35b");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    .block-container {
        background-color: rgba(0, 0, 0, 0.85);
        padding: 2rem;
        border-radius: 15px;
    }
    [data-testid="stSidebar"] {
    display: none;
}
    </style>
    """,
    unsafe_allow_html=True
)

lottie_ship = load_lottieurl(
    "https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json"
)

st_lottie(lottie_ship, height=200)
# Load your trained model
with open('titanic_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.markdown(
    "<h1 style='text-align:center; color:#FFD700;'>🚢 Titanic Survival Prediction</h1>",
    unsafe_allow_html=True
)


uploaded_file = st.file_uploader(
    "Upload a CSV file with the same columns as training data",
    type=["csv"]
)

if uploaded_file is not None:
    import pandas as pd

    data = pd.read_csv(uploaded_file)

    # Fill missing numeric values
    data["Age"] = data["Age"].fillna(data["Age"].median())
    data["Fare"] = data["Fare"].fillna(data["Fare"].median())

    # Encode categorical
    data["Sex"] = data["Sex"].map({"male": 0, "female": 1})
    data["Embarked"] = data["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    data["Embarked"] = data["Embarked"].fillna(0)

    required_cols = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']

    if all(col in data.columns for col in required_cols):

        predictions = model.predict(data[required_cols])

        data["Prediction"] = ["Survived" if p == 1 else "Not Survived" for p in predictions]

        # Remove original survival column if exists
        if "Survival" in data.columns:
            data = data.drop(columns=["Survival"])

        st.success("✅ Batch Prediction Completed")
        st.write(data)

    else:
        st.error("🚫 Uploaded CSV does NOT contain required columns.")
# Input fields
pclass = st.selectbox("Pclass", [1,2,3])
sex = st.selectbox("Sex", ["female","male"])
age = st.number_input("Age", min_value=0, max_value=100, value=25)
sibsp = st.number_input("SibSp", min_value=0, max_value=10, value=0)
parch = st.number_input("Parch", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, value=50.0)
embarked = st.selectbox("Embarked", ["S", "C", "Q"])

# Encode categorical variables the same way you did in training
sex_val = 0 if sex=="female" else 1
embarked_val = {"S":0, "C":1, "Q":2}[embarked]

# Predict button
if st.button("Predict"):
    new_passenger_df = pd.DataFrame([[pclass, sex_val, age, sibsp, parch, fare, embarked_val]],
                                    columns=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
    
    prediction = model.predict(new_passenger_df)
    if prediction[0] == 1:
        st.markdown(
             """
                <div style="
                background-color: rgba(0, 128, 0, 0.8);
                    padding: 20px;
                    border-radius: 15px;
                    text-align: center;
                    font-size: 24px;
                    font-weight: bold;
                    color: white;">
                    🎉 SURVIVED! This passenger would have lived.
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            """
                <div style="
                background-color: rgba(139, 0, 0, 0.8);
                padding: 20px;
                border-radius: 15px;
                text-align: center;
                font-size: 24px;
                font-weight: bold;
                color: white;">
                💔 DID NOT SURVIVE. This passenger would not have lived.
            </div>
            """,
            unsafe_allow_html=True
        )