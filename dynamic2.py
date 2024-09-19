import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import os
import warnings
import pyttsx3
from io import BytesIO
from sklearn.impute import KNNImputer
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
import openai
import speech_recognition as sr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from prophet import Prophet
import threading
import queue
import pyautogui
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')


def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.close()
    processed_data = output.getvalue()
    return processed_data


st.set_page_config(page_title="Data Dashboard", page_icon=":bar_chart:", layout="wide")

st.title("Advanced Data Dashboard with forecasting feature")
st.markdown('<style>div.block-container{padding-top:3rem;}</style>', unsafe_allow_html=True)


fl = st.file_uploader(":file_folder: Upload a file (CSV or Excel)", type=["csv", "xlsx", "xls"])

if fl is not None:
    if fl.name.endswith(".csv"):
        df = pd.read_csv(fl)
    elif fl.name.endswith('.xls'):
        df = pd.read_excel(fl, engine='xlrd')
    elif fl.name.endswith('.xlsx'):
        df = pd.read_excel(fl, engine='openpyxl')

    st.write("### Data Preview", df.head())

    
    st.sidebar.header("Operations")

    if st.sidebar.checkbox("Rename Columns"):
        col_to_rename = st.sidebar.selectbox("Select a column to rename", df.columns)
        new_col_name = st.sidebar.text_input("Enter new name", col_to_rename)

        if st.sidebar.button("Rename"):
            df.rename(columns={col_to_rename: new_col_name}, inplace=True)
            st.write(f"Renamed column '{col_to_rename}' to '{new_col_name}'", df.head())

    engine = pyttsx3.init()
    recognizer = sr.Recognizer()
    warnings.filterwarnings('ignore')
    speech_queue = queue.Queue()
    def speak_text_thread():
      """Thread to handle text-to-speech outside the main event loop."""
      while True:
        text = speech_queue.get()  
        if text is None:
            break  
        
        try:
            engine.say(text)
            engine.runAndWait() 
        except RuntimeError:
            continue  

      
    def speak_text(text):
     """Add text to the speech queue."""
     if not speech_queue.full():
        speech_queue.put(text)

    def recognize_speech():
     """Capture and recognize speech from the microphone."""
     with sr.Microphone() as source:
        st.write("Listening for a command...")
        audio = recognizer.listen(source)
     try:
        text = recognizer.recognize_google(audio)
        st.write(f"Recognized text: {text}")
        return text.lower()  
     except sr.UnknownValueError:
        speak_text("Sorry, I didn't understand that.")
        return None
     except sr.RequestError:
        speak_text("Sorry, I'm having trouble connecting to the speech recognition service.")
        return None
    def take_screenshot():
    
     screenshot = pyautogui.screenshot()
     screenshot.save("screenshot.png")
     st.write("Screenshot taken and saved as 'screenshot.png'.")
     speak_text("Screenshot has been taken and saved.")


    def handle_voice_command(command, df):
    
     if "summary" in command:
        st.write("### Summary of Dataset")
        speak_text("Summary for above data is given below")
        summary = df.describe()  
        st.write(summary)

        
        speak_text(f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
        speak_text(f"Here are the summary statistics: {summary.to_string()}")
        
     elif "screenshot" in command:
        take_screenshot()
        
    st.title("Voice Activated commands")
    if st.button("Click and Speak"):
            command = recognize_speech()
            if command:
                handle_voice_command(command,df)

    



    
    st.sidebar.header("Arithmetic Operations")
    col1 = st.sidebar.selectbox("Select the first column", df.columns)
    col2 = st.sidebar.selectbox("Select the second column", df.columns)
    operation = st.sidebar.selectbox("Select Operation", ["Addition", "Subtraction", "Multiplication", "Division"])
    new_col_name = st.sidebar.text_input("Enter name for the new column", "Result")

    def apply_operations(df, col1, col2, operation, new_col_name):
        if operation == "Addition":
            df[new_col_name] = df[col1] + df[col2]
        elif operation == "Subtraction":
            df[new_col_name] = df[col1] - df[col2]
        elif operation == "Multiplication":
            df[new_col_name] = df[col1] * df[col2]
        elif operation == "Division":
            df[new_col_name] = df[col1] / df[col2]
        return df

    if st.sidebar.button("Apply Operation"):
        df = apply_operations(df, col1, col2, operation, new_col_name)
        st.write(f"### Data after {operation} between {col1} and {col2}", df.head())

    
    st.sidebar.header("Custom Metrics")
    metric_formula = st.sidebar.text_input("Enter a custom formula (e.g., col1 + col2 / col3)")

    try:
        if st.sidebar.button("Apply Custom Metric"):
            df['Custom Metric'] = eval(metric_formula)
            st.write("### Data with Custom Metric", df.head())
    except Exception as e:
        st.error(f"Error in formula: {e}")
    

    def prepare_prophet_data(df, date_col, target_col):
      df[date_col] = pd.to_datetime(df[date_col], format="%d-%m-%Y")
      df_prophet = df[[date_col, target_col]].rename(columns={date_col: 'ds', target_col: 'y'})
      return df_prophet


    def prepare_ml_data(df, target_col):
    
     numeric_cols = df.select_dtypes(include=['number']).columns
     df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())  

    
     label_encoders = {}
     for col in df.select_dtypes(include=['object']).columns:
        if col != target_col:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le  

     X = df.drop(columns=[target_col])
     y = df[target_col]
     return X, y

    def make_forecast(df_prophet, periods=30):
     model = Prophet()
     model.fit(df_prophet)
    
     future = model.make_future_dataframe(periods=periods)
     forecast = model.predict(future)
     return forecast


    def make_ml_forecast(X_train, y_train, X_test, model_choice):
     if model_choice == 'Random Forest':
        model = RandomForestRegressor()
     elif model_choice == 'Linear Regression':
        model = LinearRegression()
    
     model.fit(X_train, y_train)
     predictions = model.predict(X_test)
    
     return predictions


    st.sidebar.header("Predictive Analytics")
    enable_forecasting = st.sidebar.checkbox("Enable Forecasting")

    if enable_forecasting:
        st.subheader("Predictive Analytics")
        
        model_choice = st.selectbox("Choose Model", ["Time Series Forecasting (Prophet)", "Random Forest", "Linear Regression"])
    
    
        if model_choice == "Time Series Forecasting (Prophet)":
          target_col = st.selectbox("Select the Target Column for Prediction", df.columns)
        
          date_col = st.selectbox("Select the Date Column", df.columns)
          periods = st.slider("Select Forecast Periods (Days)", min_value=1, max_value=365, value=30)
        
        
          df_prophet = prepare_prophet_data(df, date_col, target_col)
          forecast = make_forecast(df_prophet, periods)
        
          future_forecast = forecast[forecast['ds'] > df_prophet['ds'].max()]


          st.write(f"### Forecast for {periods} periods into the future")
          st.write(future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())  


          fig = px.line(future_forecast, x='ds', y='yhat', title=f"{target_col} Forecast for {periods} Days")
          st.plotly_chart(fig)
    
        elif model_choice in ["Random Forest", "Linear Regression"]:
          target_col = st.selectbox("Select the Target Column for Prediction", df.columns)
          X, y = prepare_ml_data(df, target_col)
        
        
          X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        
          predictions = make_ml_forecast(X_train, y_train, X_test, model_choice)
        
        
          st.write(f"### Predictions using {model_choice}")
          results_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
          st.write(results_df)
        
        
          fig = px.scatter(results_df, x='Actual', y='Predicted', title=f"Actual vs Predicted {target_col}")
          st.plotly_chart(fig)

    
    st.sidebar.header("Outlier Detection")
    remove_outliers = st.sidebar.checkbox("Remove Outliers (Z-score Method)")

    def remove_outliers_zscore(df):
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        df = df[(np.abs(stats.zscore(df[numeric_cols])) < 3).all(axis=1)]
        return df

    if remove_outliers:
        df = remove_outliers_zscore(df)
        st.write("### Data after outlier removal", df.head())


    st.sidebar.header("Handle Missing Values using KNN Imputation")
    n_neighbors = st.sidebar.slider("Select number of neighbors (K)", min_value=1, max_value=10, value=5)

    def knn_imputation(df, n_neighbors):
        imputer = KNNImputer(n_neighbors=n_neighbors)
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        return df

    if st.sidebar.button("Apply KNN Imputation"):
        df = knn_imputation(df, n_neighbors)
        st.write(f"### Data after KNN Imputation with K={n_neighbors}", df.head())

    
    st.sidebar.header("Chart Options")
    chart_type = st.sidebar.selectbox("Select Chart Type", ["Bar Chart", "Pie Chart", "Scatter Plot", "Line Chart"])

    if chart_type != "Pie Chart":
       x_col = st.sidebar.selectbox("Select X-axis column", df.columns)
       y_col = st.sidebar.selectbox("Select Y-axis column", df.columns)
    else:
     x_col = st.sidebar.selectbox("Select column for categories", df.columns)
     y_col = st.sidebar.selectbox("Select column for values", df.columns)

    fig = None  

    if chart_type == "Bar Chart":
     fig = px.bar(df, x=x_col, y=y_col, title=f"{chart_type}: {x_col} vs {y_col}")
    elif chart_type == "Pie Chart":
     fig = px.pie(df, names=x_col, values=y_col, title=f"{chart_type}: {x_col} Distribution")
    elif chart_type == "Scatter Plot":
     fig = px.scatter(df, x=x_col, y=y_col, title=f"{chart_type}: {x_col} vs {y_col}")
    elif chart_type == "Line Chart":
     fig = px.line(df, x=x_col, y=y_col, title=f"{chart_type}: {x_col} vs {y_col}")

    if fig:  
     st.plotly_chart(fig)


    
    st.sidebar.subheader("Download Modified File")
    st.sidebar.download_button(
        label="Download data as Excel",
        data=to_excel(df), 
        file_name='modified_data.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

    st.sidebar.subheader("Download Chart")
    st.sidebar.download_button("Download Chart as PNG", fig.to_image(format="png"), "chart.png")

else:
    st.warning("Please upload a file to begin.")
