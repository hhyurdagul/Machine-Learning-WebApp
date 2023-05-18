from backend import Timeseries, Supervised, renew_last_values

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Forecasting App")

selected = st.sidebar.radio(
    "Select the modeling type", ["Timeseries", "Supervised", "Renew Lags"]
)

if selected == "Timeseries":
    files = st.file_uploader("Select Model Files", accept_multiple_files=True)
    test_file = st.file_uploader("Upload Test File (Not necessary)")
    num = st.number_input("Number of Forecasts", min_value=1)
    if st.button("Submit"):
        ts_pipe = Timeseries(files, num, test_file)
        ts_pipe.forecast()
        col1, col2 = st.columns([3, 1])
        col1.pyplot(ts_pipe.plot_prediction())
        loss = ts_pipe.get_loss()
        if loss is not None:
            col2.write(loss)

elif selected == "Supervised":
    files = st.file_uploader("Select Model Files", accept_multiple_files=True)
    test_file = st.file_uploader("Upload Test File (Necessary)")
    num = st.number_input("Number of Forecasts", min_value=1)
    if st.button("Submit"):
        ts_pipe = Supervised(files, num, test_file)
        ts_pipe.forecast()
        col1, col2 = st.columns([3, 1])
        col1.pyplot(ts_pipe.plot_prediction())
        loss = ts_pipe.get_loss()
        col2.write(loss)

elif selected == "Renew Lags":
    col1, col2 = st.columns(2)
    last_file = col1.file_uploader("Select old last_values file")
    scaler = col2.file_uploader("Upload label scaler (If exists)")
    data = col1.file_uploader("Upload new data")
    if data is not None:
        if data.name.endswith(".csv"):
            data = pd.read_csv(data)
        else:
            data = pd.read_excel(data)
        col = col2.selectbox("Select the column", data.columns)
    else:
        col = col2.selectbox("Select the column", [])

    if st.button("Submit"):
        data = data[col]
        buffer = renew_last_values(last_file, data, scaler)
        st.download_button("Download", buffer, file_name="last_values.npy")
        buffer.close()
