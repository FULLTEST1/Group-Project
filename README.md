# Cafe Sales Prediction Dashboard

## Project Description
A streamlit dashboard that uses historical sales data and machine learning models to predict future sales.
This system is made to be used to plan production and minimise food waste

## Features

### Sales Analysis Dashboard
Interactive data visualisation of past sales:
- Total units sold
- Average daily sales
- Best-selling products
- Sales trends over time
- Product sales distribution

Users can filter the data in the sidebar by:
- Date range
- Product
- Category

### Machine Learning Prediction Dashboard
Generates predictions for future sales using historical sales data

Features include:
- Product-level demand forecasting
- Multiple machine learning models
- Forecasts for the next 28 days
- Comparison between historical and predicted sales
- Daily and weekly forecast tables

### Model Evaluation

Compare machine learning models based on performance metrics
- Accuracy
- Root Mean Square Error (RMSE)
Users can also compare different training data periods (4-8 weeks)

### Top Product Analysis
Identifies the most popular products based on past sales
- Top 3 coffee products
- Top 3 croissant products
- Sales trends over time
Users can view this for:
- Last 4 weeks
- Last 8 weeks
- All data

### Detailed Prediction View
Generate detailed predictions for a specific product using model
and training period of choice. User can download results as a CSV file

### Machine Learning Models
System allows for multiple regression algorithms to be used for forecasts:
- Random Forest Regressor
- Gradient Boosting Regressor
- Linear Regression
- Support Vector Regression

### Technologies used
- Python
- Streamlit - Interactive dashboard interface
- Pandas - Data manipulation and analysis
- Scikit-learn - Machine learning models
- Plotly - Interactive data visualisation
- PostgreSQL - Database used for storing sales data

## Running the Application
1. Clone the repository
```Bash
git clone https://github.com/FULLTEST1/Group-Project
```
3. Install dependencies
```Bash
pip install -r requirements.txt
```
3. Run the Streamlit app
```Bash
python -m streamlit run bakery_sales_dashboard.py
```

## Project Structure
```
app.py                  Main Streamlit application
requirements.txt        Project dependencies
README.md               Project documentation
```
