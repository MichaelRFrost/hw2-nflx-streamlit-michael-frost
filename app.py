import streamlit as st
import boto3
import json
import pandas as pd

st.set_page_config(page_title="HW2 NFLX Return Predictor", layout="centered")
st.title("HW2 — NFLX Next-Day Return Predictor")
st.write("This app calls a deployed SageMaker endpoint and returns a predicted next-day return.")

# Your deployed endpoint
ENDPOINT_NAME = "hw2-nflx-returns-michael-frost"
AWS_REGION = "us-east-1"

# Feature order MUST match your training order
FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Range",
    "Momentum",
    "MA_5",
    "MA_10",
    "Volatility_5"
]

st.subheader("Enter Feature Values")

vals = {}
for f in FEATURES:
    # sensible default: 0.0 (user fills in real values)
    vals[f] = st.number_input(f, value=0.0)

if st.button("Predict"):
    # Build a 1-row CSV payload in correct order, no header
    row = [vals[f] for f in FEATURES]
    csv_payload = ",".join(str(x) for x in row)

    runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)

    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="text/csv",
        Accept="application/json",
        Body=csv_payload
    )

    result = response["Body"].read().decode("utf-8")

    # Your inference.py returns a list-like string
    try:
        pred_list = json.loads(result.replace("'", '"'))
        pred = float(pred_list[0])
    except:
        pred = float(result.strip("[] ").split(",")[0])

    st.success(f"Predicted next-day return: {pred:.6f}")
    st.caption("Example: 0.01 = +1% predicted return")
