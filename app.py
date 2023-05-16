from backend import Timeseries

import streamlit as st

st.set_page_config(page_title="Forecasting App")

files = st.file_uploader("Select Model Files", accept_multiple_files=True)
test_file = st.file_uploader("Upload Test File (Not necessary)")
num = st.number_input("Number of Forecasts", min_value=1)
if st.button("Submit"):
    ts_pipe = Timeseries(files, num, test_file)
    ts_pipe.forecast()
    col1, col2 = st.columns([3,1])
    col1.pyplot(ts_pipe.plot_prediction())
    loss = ts_pipe.get_loss()
    if loss is not None:
        col2.write(loss)
