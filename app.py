import gradio as gr
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
df = pd.read_csv("Social_Network_Ads.csv")

X = df.drop("Purchased", axis=1)
y = df["Purchased"]

X = X.drop("User ID", axis=1)
X["Gender"] = X["Gender"].map({"Male": 1, "Female": 0})
X["Salary_per_Age"] = X["EstimatedSalary"] / X["Age"]

model = Pipeline([
    ("scaler", StandardScaler()),("classifier", LogisticRegression(max_iter=500))
])

model.fit(X, y)

def predict_purchase(age, gender, salary):
    gender_val = 1 if gender == "Male" else 0
    salary_per_age = salary / age

    input_df = pd.DataFrame(
        [[gender_val, age, salary, salary_per_age]],
        columns=["Gender", "Age", "EstimatedSalary", "Salary_per_Age"]
    )

    prediction = model.predict(input_df)[0]

    return "Likely to Purchase" if prediction == 1 else "Not Likely to Purchase"

interface = gr.Interface(
    fn=predict_purchase,
    inputs=[
        gr.Number(label="Age"),
        gr.Radio(["Male", "Female"], label="Gender"),
        gr.Number(label="Estimated Salary")
    ],
    outputs="text",
    title="Social Network Ads Purchase Prediction",
    description="Predict whether a user will purchase a product based on demographics."
)

interface.launch()
