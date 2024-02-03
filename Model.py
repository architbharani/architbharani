import datetime
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from plotly import graph_objs as go

from LSTM import LSTMModel
from sklearn.preprocessing import MinMaxScaler,StandardScaler

st.title("Energy Prediction")

min_date = datetime.date(2015,7,10 )  # Minimum allowed date
max_date = datetime.date(2022, 11, 30)  # Maximum allowed date

default_date = datetime.date(2022, 1, 1)

selected_date = st.date_input(
    "Select a date within the specified range:",
    min_value=min_date,
    max_value=max_date,
    value=default_date
)

selected_date_str = selected_date.strftime("%Y-%m-%d")

data = pd.read_excel("15-17(2).xlsx", parse_dates=['Date'])
df = pd.read_excel("15-17(2).xlsx", index_col='Date', parse_dates=['Date'])

def create_features(df):
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['dayofyear'] = df.index.dayofyear
    return df

df = df.iloc[:,:3]

df = create_features(df)
df = df.iloc[:,:7]

def sin_cos(df):
    # Sine and cosine transformations for day of week
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    # Sine and cosine transformations for day of year
    df['doy_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)

    # Sine and cosine transformations for month
    df['m_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['m_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Sine and cosine transformations for quarter
    df['q_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
    df['q_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)

    return df

df = sin_cos(df)
df = df.drop(['month', 'dayofyear', 'quarter','dayofweek'], axis=1)

if 'Future' not in st.session_state:
    st.session_state.value = 0

# Create the slider
Future = st.slider('Select a value', 1, 15)
model = LSTMModel(input_dim=11, hidden_dim=64, num_layers=2, output_dim=11)
model.load_state_dict(torch.load('LSTM.pth'))
scaler = MinMaxScaler()

# Check if the slider value has changed
if Future != st.session_state.value:
    indices_of_date = data.loc[data['Date'] == selected_date_str].index
    index_of_date = indices_of_date[0].item()
    start_index = max(0, index_of_date -100)

    # Ensure the start index is not negative
    df = scaler.fit_transform(df)
    x_input = df[start_index:index_of_date]
    x_input = torch.tensor(x_input)
    x_input = x_input.view(1, 100, 11)
    x_input = torch.tensor(x_input).float()

    model.eval()
    future_output = []
    i = 0
    while (i < Future):
        with torch.no_grad():
            yhat = model(x_input)
        future_output.extend(yhat.tolist())
        x_input = torch.cat([x_input[:, 1:, :], yhat.view(1, 1, -1)], dim=1)
        i += 1

    future_output = scaler.inverse_transform(future_output)
    future_df = pd.DataFrame(future_output)

    df = scaler.inverse_transform(df)

    date_index = pd.date_range(selected_date_str, periods=Future, freq='D')
    df_result = pd.DataFrame(index=date_index)
    actual = data.iloc[index_of_date:index_of_date + Future, 1]
    actual.index = pd.date_range(selected_date_str, periods=Future, freq='D')
    future_df.index = pd.date_range(selected_date_str, periods=Future, freq='D')
    df_result = pd.concat([df_result, actual, future_df.iloc[:, 0]], axis=1)
    df_result.columns = ['Actual', 'Predicted']
    st.write("Final Output DataFrame:")
    st.write(df_result)


    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Line(x=data['Date'], y=data['Electricity'], name='Time Series'))
        # fig.add_trace(go.Line(x=df_result.index, y=df_result['Predicted'], name='Predictions'))
        fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data()


    def plot_raw_data1():
        # Assuming df_result_index is a pandas DataFrame or Series
        filtered_data = data[data['Date'].isin(df_result.index)]

        fig = go.Figure()
        fig.add_trace(go.Line(x=filtered_data['Date'], y=filtered_data['Electricity'], name='Time Series'))
        fig.add_trace(go.Line(x=df_result.index, y=df_result['Predicted'], name='Predictions'))
        fig.layout.update(title_text="Actual vs Predicted", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)


    # Call the function
    plot_raw_data1()


st.session_state.value = Future

