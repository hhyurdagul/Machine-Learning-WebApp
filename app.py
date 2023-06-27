from backend import (
    Timeseries,
    Supervised,
    renew_last_values,
    renew_lookback_values,
    renew_seasonal_lookback_values,
)


import streamlit as st
import pandas as pd

st.set_page_config(page_title="Forecasting App")

selected = st.sidebar.radio(
    "Select the modeling type",
    [
        "Timeseries",
        "Supervised",
        "Renew Lags",
        "Renew Lookback",
    ],
)

if selected == "Timeseries":
    test_file = st.file_uploader("Upload Test File (Not necessary)")
    num = st.number_input("Number of Forecasts", min_value=1)
    if st.button("Submit"):
        ts_pipe = Timeseries(num, test_file)
        ts_pipe.forecast()
        col1, col2 = st.columns([3, 1])
        df, fig = ts_pipe.plot_prediction()
        col1.pyplot(fig)
        col2.dataframe(df)
        loss = ts_pipe.get_loss()
        if loss is not None:
            st.write(loss)

elif selected == "Supervised":
    test_file = st.file_uploader("Upload Test File (Necessary)")
    num = st.number_input("Number of Forecasts", min_value=1)
    if st.button("Submit"):
        sv_pipe = Supervised(num, test_file)
        sv_pipe.forecast()
        col1, col2 = st.columns([3, 1])
        df, fig = sv_pipe.plot_prediction()
        col1.pyplot(fig)
        col2.dataframe(df)
        loss = sv_pipe.get_loss()
        st.write(loss)

elif selected == "Renew Lags":
    col1, col2 = st.columns(2)
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
        renew_last_values(data)
        st.write("Renewed Lags")

elif selected == "Renew Lookback":
    col1, col2 = st.columns(2)
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
        renew_lookback_values(data.copy())
        renew_seasonal_lookback_values(data.copy())
        st.write("Renewed Lookbacks")
