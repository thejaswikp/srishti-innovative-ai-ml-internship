import streamlit as st
import joblib
from sklearn.datasets import load_iris

# Load model
model = joblib.load("iris_model.pkl")

iris = load_iris()

st.title("🌸 Iris Flower Prediction")

st.write("Enter flower measurements")

sl = st.slider("Sepal Length", 4.0, 8.0, 5.1)
sw = st.slider("Sepal Width", 2.0, 4.5, 3.5)
pl = st.slider("Petal Length", 1.0, 7.0, 1.4)
pw = st.slider("Petal Width", 0.1, 2.5, 0.2)

prediction = model.predict([[sl, sw, pl, pw]])

species = iris.target_names[prediction[0]]

st.success(f"Predicted Species: {species}")