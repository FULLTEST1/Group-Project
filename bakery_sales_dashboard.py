# pip install streamlit pandas psycopg2 plotly scikit-learn numpy
# streamlit run bakery_sales_dashboard.py (cd to folder first)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import psycopg2
from datetime import datetime, timedelta
from io import StringIO
import re
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Cafe Sales Dashboard", page_icon="☕", layout="wide")

# ============ DATABASE FUNCTIONS ============

@st.cache_resource
def get_connection():
    # Establishes connection to the database
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="cafe_sales_db",
            user="postgres",
            password="password123"
        )
        return conn
    except Exception as e:
        st.error(f"Cannot connect to database: {e}")
        return None

def load_sales_data():
    # Load sales data from the database
    conn = get_connection()
    if conn is None:
        return None
    
    query = """
    SELECT 
        ds.sale_date,
        p.product_name,
        p.category,
        ds.units_sold
    FROM daily_sales ds
    JOIN products p ON ds.product_id = p.product_id
    ORDER BY ds.sale_date DESC
    """
    
    try:
        df = pd.read_sql(query, conn)
        df['sale_date'] = pd.to_datetime(df['sale_date'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def detect_date_column(df):
    date_keywords = ['date', 'day', 'time', 'when', 'period']
    
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in date_keywords):
            return col
        
        try:
            pd.to_datetime(df[col].iloc[0])
            return col
        except:
            continue
    
    return df.columns[0]

def parse_date_flexible(date_str):
    date_formats = [
        '%d/%m/%Y', '%m/%d/%Y', '%Y-%m-%d', '%d-%m-%Y',
        '%Y/%m/%d', '%d.%m.%Y', '%d %B %Y', '%d %b %Y'
    ]
    
    for fmt in date_formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except:
            continue
    
    try:
        return pd.to_datetime(date_str)
    except:
        return None

def ensure_product_exists(product_name, category='coffee'):

    # Checks if a product exists in the database. If not then it creates it
    conn = get_connection()
    if conn is None:
        return None
    
    cur = conn.cursor()
    
    try:
        cur.execute("""
            INSERT INTO products (product_name, category)
            VALUES (%s, %s)
            ON CONFLICT (product_name) DO NOTHING
            RETURNING product_id;
        """, (product_name, category))
        
        result = cur.fetchone()
        if result:
            product_id = result[0]
        else:
            cur.execute("SELECT product_id FROM products WHERE product_name = %s", (product_name,))
            product_id = cur.fetchone()[0]
        
        conn.commit()
        cur.close()
        return product_id
    except Exception as e:
        conn.rollback()
        cur.close()
        st.error(f"Error creating product: {e}")
        return None

def process_flexible_csv(df, date_col, product_cols, product_info):
    # Inserts data from a CSV into the database
    conn = get_connection()
    if conn is None:
        return False
    
    cur = conn.cursor()
    
    try:
        success_count = 0
        error_count = 0
        
        for idx, row in df.iterrows():
            try:
                sale_date = parse_date_flexible(str(row[date_col]))
                
                if sale_date is None:
                    error_count += 1
                    continue
                
                for product_col in product_cols:
                    product_name = product_info[product_col]['name']
                    category = product_info[product_col]['category']
                    units_sold = row[product_col]
                    
                    if pd.isna(units_sold) or units_sold == '' or units_sold == 0:
                        continue
                    
                    units_sold = int(float(units_sold))
                    
                    product_id = ensure_product_exists(product_name, category)
                    
                    if product_id:
                        cur.execute("""
                            INSERT INTO daily_sales (sale_date, product_id, units_sold)
                            VALUES (%s, %s, %s)
                            ON CONFLICT (sale_date, product_id)
                            DO UPDATE SET units_sold = EXCLUDED.units_sold;
                        """, (sale_date, product_id, units_sold))
                        success_count += 1
                
            except Exception as e:
                error_count += 1
                continue
        
        conn.commit()
        cur.close()
        
        if error_count > 0:
            st.warning(f"⚠️ Imported {success_count} records, {error_count} rows had errors")
        else:
            st.success(f"✅ Successfully imported {success_count} records!")
        
        return True
        
    except Exception as e:
        conn.rollback()
        cur.close()
        st.error(f"Error processing CSV: {e}")
        return False

# ============ ML FUNCTIONS ============

def prepare_ml_features(df, product_name, training_weeks=4):
    """
    Prepares for the model to predict sales of individual product

    Steps:
    1. Filters data for the given product
    2. Sorts by date
    3. Adds features such as day of the week/month/year for seasonal predictions
    4. Adds lag features for past sales
    5. Adds rolling statistics
    6. Removes rows with missing values
    7. Check for enough data
    8. Selects training data

    """
    product_data = df[df['product_name'] == product_name].copy()
    product_data = product_data.sort_values('sale_date')
    
    product_data['day_of_week'] = product_data['sale_date'].dt.dayofweek
    product_data['day_of_month'] = product_data['sale_date'].dt.day
    product_data['week_of_year'] = product_data['sale_date'].dt.isocalendar().week
    product_data['month'] = product_data['sale_date'].dt.month
    
    for lag in [1, 2, 3, 7]:
        product_data[f'lag_{lag}'] = product_data['units_sold'].shift(lag)
    
    product_data['rolling_mean_7'] = product_data['units_sold'].rolling(window=7, min_periods=1).mean()
    product_data['rolling_std_7'] = product_data['units_sold'].rolling(window=7, min_periods=1).std()
    product_data['rolling_mean_14'] = product_data['units_sold'].rolling(window=14, min_periods=1).mean()
    
    product_data = product_data.dropna()
    
    if len(product_data) < training_weeks * 7:
        return None, None, None
    
    train_data = product_data.tail(training_weeks * 7)
    
    feature_cols = ['day_of_week', 'day_of_month', 'week_of_year', 'month',
                    'lag_1', 'lag_2', 'lag_3', 'lag_7',
                    'rolling_mean_7', 'rolling_std_7', 'rolling_mean_14']
    
    X = train_data[feature_cols]
    y = train_data['units_sold']
    
    return X, y, feature_cols

def train_models(X, y):

    """
    Train models and evaluate their performance

    Steps:
    1. Split data into train/test
    2. Define models e.g Random Forest, Gradient Boosting
    3. Train each model on training data
    4. Predict the sales on test set
    5. Calculate evaluation metrics e.g RMSE
    6. Skips models that fail to train to prevent crashing
    7. Return trained models and their performance scores
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    models = {
        'Random Forest': RandomForestRegressor(),
        'Gradient Boosting': GradientBoostingRegressor(),
        'Linear Regression': LinearRegression(),
        'Support Vector Regression': SVR()
    }
    
    trained_models = {}
    model_scores = {}
    
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)
            
            trained_models[name] = model
            model_scores[name] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'Accuracy': max(0, (1 - mae / y.mean()) * 100)
            }
        except Exception as e:
            pass
    
    return trained_models, model_scores

def predict_future_sales(model, product_data, feature_cols, days_ahead=28):

    """
    Predict the future sales of a product

    Steps:
    1. Copy and sort data to preserve time order
    2. Loop over the number of days being predicted
    3. For each day
        a. Compute the date-based features (day/week/month)
        b. Compute lag features from past sales
        c. Combine these features into single-row DataFrame
        d. Predict sales using model
        e. Don't allow negatives
        f. Store prediction
        g. Add predicted day to historical data for next iteration
    4. Return a DataFrame of predictions
    """
    product_data = product_data.copy()
    product_data = product_data.sort_values("sale_date")

    last_date = product_data["sale_date"].max()

    predictions = []

    for i in range(days_ahead):

        future_date = last_date + timedelta(days=i+1)

        # ----- Date features -----
        day_of_week = future_date.weekday()
        day_of_month = future_date.day
        week_of_year = future_date.isocalendar()[1]
        month = future_date.month

        # ----- Lag features -----
        lag_1 = product_data["units_sold"].iloc[-1]
        lag_2 = product_data["units_sold"].iloc[-2] if len(product_data) >= 2 else lag_1
        lag_3 = product_data["units_sold"].iloc[-3] if len(product_data) >= 3 else lag_1
        lag_7 = product_data["units_sold"].iloc[-7] if len(product_data) >= 7 else lag_1

        # ----- Rolling features -----
        rolling_mean_7 = product_data["units_sold"].tail(7).mean()
        rolling_std_7 = product_data["units_sold"].tail(7).std()
        rolling_mean_14 = product_data["units_sold"].tail(14).mean()

        if pd.isna(rolling_std_7):
            rolling_std_7 = 0

        features = pd.DataFrame([{
            "day_of_week": day_of_week,
            "day_of_month": day_of_month,
            "week_of_year": week_of_year,
            "month": month,
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_3": lag_3,
            "lag_7": lag_7,
            "rolling_mean_7": rolling_mean_7,
            "rolling_std_7": rolling_std_7,
            "rolling_mean_14": rolling_mean_14
        }])

        prediction = model.predict(features)[0]

        prediction = max(0, prediction)

        predictions.append({
            "date": future_date,
            "predicted_sales": prediction
        })

        # Update dataset so next prediction can use it as lag
        product_data = pd.concat([
            product_data,
            pd.DataFrame({
                "sale_date": [future_date],
                "units_sold": [prediction]
            })
        ])

    predictions_df = pd.DataFrame(predictions)

    return predictions_df

def get_summary_stats(df):

    """
    Calculates basic summary statistics

    Statistics calculated:
    - Total sales for all products and dates
    - Average daily sales
    - Best-selling product and its total sales
    - Date range in the dataset
    - Number of unique days with sales
    """
    stats = {}
    
    stats['total_sales'] = df['units_sold'].sum()
    stats['avg_daily'] = df.groupby('sale_date')['units_sold'].sum().mean()
    
    product_sales = df.groupby('product_name')['units_sold'].sum()
    stats['best_product'] = product_sales.idxmax()
    stats['best_product_sales'] = product_sales.max()
    
    stats['date_range'] = f"{df['sale_date'].min().strftime('%Y-%m-%d')} to {df['sale_date'].max().strftime('%Y-%m-%d')}"
    stats['total_days'] = df['sale_date'].nunique()
    
    return stats

# ============ UPLOAD PAGE ============

def upload_page():
    # ----- Section header -----
    st.title("☕ Cafe Sales Dashboard")
    st.markdown("---")
    
    st.header("📤 Upload Sales Data")
    st.write("Upload your CSV file - I'll automatically detect the format!")
    
    # ----- File Uploader -----
    # Lets user upload ONLY .csv files
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload any CSV with a date column and product sales columns"
    )
    
    # Shows example of CSV file format
    with st.expander("📋 Supported CSV Formats"):
        st.write("**Format 1: Multiple Products**")
        example1 = pd.DataFrame({
            'Date': ['01/03/2025', '02/03/2025'],
            'Cappuccino': [82, 57],
            'Americano': [100, 103],
            'Plain Croissant': [25, 30]
        })
        st.dataframe(example1)
        
        st.info("""
        **Flexible Features:**
        - Any date format (DD/MM/YYYY, YYYY-MM-DD, etc.)
        - Any date column name (Date, Day, Time, etc.)
        - Any number of product columns
        - Auto-detects categories (coffee/croissants)
        - Customize product names before import
        """)
    
    # If file was uploaded reads it into DataFrame
    # Previews top 10 rows and row/column count for confirmation
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.write("**📊 File Preview:**")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.write(f"**Rows:** {len(df)} | **Columns:** {len(df.columns)}")
            
            st.markdown("---")
            st.subheader("🎯 Column Mapping")
            
            # ----- Detect Date Column -----
            # Detects which columns contain dates
            # User can confirm or change the detected column
            date_col = detect_date_column(df)
            st.info(f"📅 Detected date column: **{date_col}**")
            
            date_col_selected = st.selectbox(
                "Confirm or change date column:",
                options=df.columns,
                index=list(df.columns).index(date_col)
            )
            
            # Identify all other columns
            other_cols = [col for col in df.columns if col != date_col_selected]
            
            if len(other_cols) == 0:
                st.error("❌ No product columns found!")
                return
            
            st.write(f"**🛍️ Found {len(other_cols)} product column(s):**")
            
            # User can pick which columns are product sales
            product_cols = st.multiselect(
                "Select which columns contain sales data:",
                options=other_cols,
                default=other_cols
            )
            
            if not product_cols:
                st.warning("⚠️ Please select at least one product column")
                return
            
            st.write("**📝 Customize product names and categories:**")
            
            # ===== Product Name and Categories =====
            product_info = {}
            
            for product in product_cols:
                st.markdown(f"**{product}**")
                col1, col2 = st.columns([2, 1])
                
                # Lets you rename products
                with col1:
                    custom_name = st.text_input(
                        "Product name:",
                        value=product,
                        key=f"name_{product}",
                        help="Change this if the column name doesn't match the actual product name"
                    )
                # Detects category
                with col2:
                    product_lower = product.lower()
                    
                    if any(word in product_lower for word in ['croissant', 'croissants', 'pastry', 'pastries', 'baked', 'bread']):
                        default_cat = 'croissants'
                    elif any(word in product_lower for word in ['coffee', 'espresso', 'cappuccino', 'latte', 'americano', 'mocha', 'macchiato']):
                        default_cat = 'coffee'
                    else:
                        default_cat = 'coffee'
                    
                    category = st.selectbox(
                        "Category:",
                        options=['coffee', 'croissants'],
                        index=0 if default_cat == 'coffee' else 1,
                        key=f"cat_{product}"
                    )
                # Stores in dictionary for database import
                product_info[product] = {
                    'name': custom_name,
                    'category': category
                }
            
            st.markdown("---")
            
            col1, col2, col3 = st.columns([1, 1, 2])

            """
            Import to Database

            Steps:
            1. User clicks button to import CSV
            2. process_flexible_csv inserts it into database
            3. Shows loading spinner during processing
            4. Shows balloons when successful
            5. session_state.data_loaded is set, used to display dashboard
            6. Refreshes page
            """
            with col1:
                if st.button("📥 Import to Database", type="primary", use_container_width=True):
                    with st.spinner("Importing to database..."):
                        if process_flexible_csv(df, date_col_selected, product_cols, product_info):
                            st.balloons()
                            st.session_state.data_loaded = True
                            st.rerun()
            
            # ===== Cancel Button =====
            with col2:
                if st.button("🔄 Cancel", use_container_width=True):
                    st.rerun()
                    
        except Exception as e:
            st.error(f"Error reading file: {e}")
    
    st.markdown("---")
    
    # ===== Database Summary =====
    conn = get_connection()
    if conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM daily_sales;")
        record_count = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(DISTINCT product_id) FROM daily_sales;")
        product_count = cur.fetchone()[0]
        
        cur.close()
        
        if record_count > 0:
            st.success(f"💾 Database contains **{record_count}** sales records across **{product_count}** products")
            
            # ===== Navigate Buttons =====
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button("📊 View Dashboard", use_container_width=True):
                    st.session_state.data_loaded = True
                    st.rerun()
            
            with col2:
                if st.button("🗑️ Manage Data", use_container_width=True):
                    st.session_state.show_delete = True
                    st.rerun()
            
            # ===== Data Deletion Section =====
            if st.session_state.get('show_delete', False):
                st.markdown("---")
                st.subheader("🗑️ Delete Data")
                
                delete_option = st.radio(
                    "What would you like to delete?",
                    ["Specific Date Range", "Specific Products", "Everything"],
                    horizontal=True
                )
                
                if delete_option == "Specific Date Range":
                    st.write("Delete all sales data within a date range:")
                    
                    cur = conn.cursor()
                    cur.execute("SELECT MIN(sale_date), MAX(sale_date) FROM daily_sales;")
                    min_d, max_d = cur.fetchone()
                    cur.close()
                    
                    del_date_range = st.date_input(
                        "Select dates to delete:",
                        value=(min_d, max_d),
                        min_value=min_d,
                        max_value=max_d
                    )
                    
                    if len(del_date_range) == 2:
                        cur = conn.cursor()
                        cur.execute("""
                            SELECT COUNT(*) FROM daily_sales 
                            WHERE sale_date >= %s AND sale_date <= %s
                        """, (del_date_range[0], del_date_range[1]))
                        affected_records = cur.fetchone()[0]
                        cur.close()
                        
                        st.warning(f"⚠️ This will delete **{affected_records}** sales records")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("🗑️ Confirm Delete", type="primary", use_container_width=True):
                                cur = conn.cursor()
                                cur.execute("""
                                    DELETE FROM daily_sales 
                                    WHERE sale_date >= %s AND sale_date <= %s
                                """, (del_date_range[0], del_date_range[1]))
                                conn.commit()
                                cur.close()
                                st.success(f"✅ Deleted {affected_records} records!")
                                st.session_state.show_delete = False
                                st.rerun()
                        
                        with col2:
                            if st.button("❌ Cancel", use_container_width=True):
                                st.session_state.show_delete = False
                                st.rerun()
                
                elif delete_option == "Specific Products":
                    st.write("Delete all sales data for specific products:")
                    
                    cur = conn.cursor()
                    cur.execute("SELECT product_id, product_name FROM products ORDER BY product_name;")
                    products = cur.fetchall()
                    cur.close()
                    
                    product_dict = {name: pid for pid, name in products}
                    
                    products_to_delete = st.multiselect(
                        "Select products to delete:",
                        options=list(product_dict.keys())
                    )
                    
                    if products_to_delete:
                        product_ids = [product_dict[name] for name in products_to_delete]
                        
                        cur = conn.cursor()
                        cur.execute("""
                            SELECT COUNT(*) FROM daily_sales 
                            WHERE product_id = ANY(%s)
                        """, (product_ids,))
                        affected_records = cur.fetchone()[0]
                        cur.close()
                        
                        st.warning(f"⚠️ This will delete **{affected_records}** sales records")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("🗑️ Confirm Delete", type="primary", use_container_width=True):
                                cur = conn.cursor()
                                cur.execute("""
                                    DELETE FROM daily_sales 
                                    WHERE product_id = ANY(%s)
                                """, (product_ids,))
                                conn.commit()
                                
                                cur.execute("""
                                    DELETE FROM products 
                                    WHERE product_id = ANY(%s)
                                """, (product_ids,))
                                conn.commit()
                                cur.close()
                                
                                st.success(f"✅ Deleted {affected_records} records!")
                                st.session_state.show_delete = False
                                st.rerun()
                        
                        with col2:
                            if st.button("❌ Cancel", use_container_width=True):
                                st.session_state.show_delete = False
                                st.rerun()
                
                elif delete_option == "Everything":
                    st.error("⚠️ **WARNING: This will delete ALL data!**")
                    
                    confirm_text = st.text_input("Type 'DELETE ALL' to confirm:")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🗑️ Delete Everything", type="primary", use_container_width=True, disabled=(confirm_text != "DELETE ALL")):
                            cur = conn.cursor()
                            cur.execute("TRUNCATE TABLE daily_sales CASCADE;")
                            cur.execute("TRUNCATE TABLE products CASCADE;")
                            conn.commit()
                            cur.close()
                            st.success("✅ All data deleted!")
                            st.session_state.show_delete = False
                            st.rerun()
                    
                    with col2:
                        if st.button("❌ Cancel", use_container_width=True):
                            st.session_state.show_delete = False
                            st.rerun()

# ============ DASHBOARD PAGES ============

def apply_plotly_theme(fig, template, theme_mode, height=400):
    """
    Apply background and font colors to a Plotly figure based on theme_mode.
    Works for line, bar, and pie charts.
    """

    # ===== Theme color Selection =====
    if theme_mode == "Light Mode":
        bg_color = "white"
        font_color = "black"
        slider_bg = "white"
    else:
        bg_color = "#0E1117"
        font_color = "white"
        slider_bg = "#0E1117"
    
    # General layout of chart
    fig.update_layout(
        template=template,
        # colors set by theme
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=font_color),
        title=dict(font=dict(color=font_color)),  # Figure title
        legend=dict(
            font=dict(color=font_color),
            title=dict(font=dict(color=font_color))
        ),
        # Adds zoom slider under charts
        xaxis=dict(
            rangeslider=dict(
                visible=True,
                bgcolor=slider_bg
            ),
            title=dict(font=dict(color=font_color)),
            tickfont=dict(color=font_color)
        ),
        yaxis=dict(
            title=dict(font=dict(color=font_color)),
            tickfont=dict(color=font_color)
        ),
        height=height
    )

    # Special handling for pie charts
    for trace in fig.data:
        if getattr(trace, "type", None) == "pie":
            fig.update_traces(
                textfont=dict(color=font_color),
                marker=dict(line=dict(color=bg_color, width=1))
            )
    return fig

def apply_streamlit_theme(theme_mode):
    # ===== Theme color Selection =====
    if theme_mode == "Light Mode":
        text = "#000000"
        bg = "#FFFFFF"
        sidebar_bg = "#F5F5F5"
        border = "#DDDDDD"
    else:
        text = "#FFFFFF"
        bg = "#0E1117"
        sidebar_bg = "#1C1E26"
        border = "#555"
    
    # CSS injection for custom styling using Markdown
    st.markdown(f"""
    <style>

    /* Main app text */
    html, body, [class*="css"] {{
        color: {text};
    }}

    /* Headers */
    h1, h2, h3, h4, h5, h6 {{
        color: {text} !important;
    }}

    /* Paragraphs / markdown */
    p, span {{
        color: {text};
    }}

    /* Widget labels */
    label {{
        color: {text} !important;
    }}

    /* General Streamlit text containers */
    [data-testid="stMarkdownContainer"],
    [data-testid="stText"] {{
        color: {text};
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {sidebar_bg};
        color: {text};
    }}

    section[data-testid="stSidebar"] * {{
        color: {text} !important;
    }}

    /* Metrics */
    [data-testid="stMetricValue"] {{
        color: {text};
    }}

    [data-testid="stMetricLabel"] {{
        color: {text};
    }}

    /* Widget labels */
    label {{
        color: {text} !important;
    }}

    /* Expander text */
    .streamlit-expanderHeader {{
        color: {text};
    }}

    /* Tabs */
    button[role="tab"] {{
        color: {text};
    }}

    /* Dataframes */
    [data-testid="stDataFrame"] {{
        color: {text};
    }}

    table {{
        color: {text};
        border: 1px solid {border};
    }}

    th, td {{
        color: {text};
    }}

    /* Buttons */
    button {{
        color: {text} !important;
    }}

    </style>
    """, unsafe_allow_html=True)

def apply_global_theme(theme_mode):
    # ===== Theme color Selection =====
    if theme_mode == "Light Mode":
        bg = "#FFFFFF"
        secondary_bg = "#F5F5F5"
        text = "#000000"
        border = "#CCCCCC"
        input_bg = "#FFFFFF"
    else:
        bg = "#0E1117"
        secondary_bg = "#1C1E26"
        text = "#FFFFFF"
        border = "#555"
        input_bg = "#1C1E26"
    

    st.markdown(f"""
    <style>
    /* Main app background */
    .stApp {{
        background-color: {bg};
        color: {text};
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {secondary_bg};
        color: {text};
    }}

    /* Headers */
    h1, h2, h3, h4, h5, h6 {{
        color: {text};
    }}

    /* Buttons */
    button {{
        background-color: {secondary_bg};
        color: {text};
        border: 1px solid {border};
    }}

    /* DataFrames */
    [data-testid="stDataFrame"] * {{
        color: {text} !important;
        background-color: {bg} !important;
    }}

    /* Tables */
    table {{
        width:100%;
        border-collapse: collapse;
        background-color: {bg};
        color: {text};
    }}

    table th, table td {{
        border: 1px solid {border};
        padding: 8px;
    }}

    /* Expander headers */
    div[data-testid="stExpander"] > div {{
        background-color: {secondary_bg};
        color: {text};
    }}

    /* Inputs, sliders, selectboxes, radio buttons, checkboxes */
    div[data-baseweb] {{
        color: {text} !important;
        background-color: {input_bg} !important;
    }}

    /* Slider handles and tracks */
    div[data-testid="stSlider"] .rc-slider-track,
    div[data-testid="stSlider"] .rc-slider-handle {{
        background-color: {text} !important;
        border-color: {text} !important;
    }}

    /* File uploader background */
    div[data-testid="stFileUploader"] {{
        background-color: {secondary_bg};
        color: {text};
    }}

    /* Text inputs / text areas */
    input, textarea {{
        background-color: {input_bg} !important;
        color: {text} !important;
        border: 1px solid {border};
    }}

    /* Download button */
    button[kind="secondary"] {{
        background-color: {secondary_bg};
        color: {text};
        border: 1px solid {border};
    }}

    /* Sidebar widgets */
    section[data-testid="stSidebar"] div[data-baseweb] {{
        background-color: {secondary_bg} !important;
        color: {text} !important;
    }}
    /* ===== EXPANDER THEME FIX ===== */

    /* Expander container */
    div[data-testid="stExpander"] details {{
        background-color: {secondary_bg} !important;
        border: 1px solid {border} !important;
        border-radius: 6px;
    }}

    /* Expander header */
    div[data-testid="stExpander"] summary {{
        background-color: {secondary_bg} !important;
        color: {text} !important;
        padding: 6px;
    }}

    /* Expander header when open */
    div[data-testid="stExpander"] details[open] summary {{
        background-color: {secondary_bg} !important;
        color: {text} !important;
    }}

    /* Expander content */
    div[data-testid="stExpander"] details > div {{
        background-color: {bg} !important;
        color: {text} !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# ===== Theme Selection =====
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Dark Mode"  # default

with st.sidebar:
    st.markdown("## 🎨 Theme")
    theme_choice = st.radio(
        "Select theme:",
        ["Light Mode", "Dark Mode"],
        index=0 if st.session_state.theme_mode == "Light Mode" else 1
    )
    st.session_state.theme_mode = theme_choice

# Apply theme immediately
apply_global_theme(st.session_state.theme_mode)
apply_streamlit_theme(st.session_state.theme_mode)

def analysis_dashboard():
    """
    Historical analysis dashboard
    
    Displays cafe sales data stored in the database and allows users to:
    - Filter by date range, product and category
    - View summary metrics
    - Explore interactive sales visualisations
    """

    # ===== Theme & template =====
    theme_mode = st.session_state.get("theme_mode", "Light Mode")
    template = "plotly_white" if theme_mode == "Light Mode" else "plotly_dark"

    # Product colors
    category_colors = {
        'Americano': "#FF5733",    # bright red-orange
        'Cappuccino': '#33C1FF',   # bright blue
        'Croissants': "#F460AA"    # pinkish
    }

    st.title("☕ Sales Analysis Dashboard")
    
    df = load_sales_data()
    # Error handling
    if df is None or df.empty:
        st.warning("⚠️ No data in database")
        return

    # ===== Sidebar filters =====
    # Date Range Filter
    st.sidebar.header("📊 Filters")
    min_date = df['sale_date'].min()
    max_date = df['sale_date'].max()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    # Product Filter
    products = ['All Products'] + sorted(df['product_name'].unique())
    selected_product = st.sidebar.selectbox("Select Product", products)
    # Category FIlter
    categories = ['All Categories'] + sorted(df['category'].unique())
    selected_category = st.sidebar.selectbox("Select Category", categories)
    
    # ===== Apply Filters =====
    if len(date_range) == 2:
        mask = (df['sale_date'] >= pd.Timestamp(date_range[0])) & (df['sale_date'] <= pd.Timestamp(date_range[1]))
        filtered_df = df[mask]
    else:
        filtered_df = df

    # Only filter if the user selects a product
    if selected_product != 'All Products':
        filtered_df = filtered_df[filtered_df['product_name'] == selected_product]
    
    # Only filter if the user selects a category
    if selected_category != 'All Categories':
        filtered_df = filtered_df[filtered_df['category'] == selected_category]
    
    # Empty filter result warning
    if filtered_df.empty:
        st.warning("No data matches the selected filters")
        return

    # ===== Key Metrics =====
    stats = get_summary_stats(filtered_df)
    st.header("📈 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Units Sold", f"{stats['total_sales']:,.0f}")
    col2.metric("Average Daily Sales", f"{stats['avg_daily']:.1f}")
    col3.metric("Best Seller", stats['best_product'], delta=f"{stats['best_product_sales']:,.0f} units")
    col4.metric("Days Tracked", f"{stats['total_days']}")
    st.markdown("---")

    # ===== Sales Visualizations =====
    st.header("📊 Sales Visualizations")
    
    daily_sales = filtered_df.groupby(['sale_date', 'product_name', 'category'])['units_sold'].sum().reset_index()

    # Build a consistent color map for all charts
    color_map = {
        row['product_name']: category_colors.get(row['product_name'], '#888')  # default gray
        for _, row in filtered_df[['product_name']].drop_duplicates().iterrows()
    }

    # ===== Daily Sales Line Chart =====
    # Creates a time-series chart - Date vs Units Sold
    fig_daily = px.line(
        daily_sales,
        x='sale_date',
        y='units_sold',
        color='product_name',
        line_group='category',
        color_discrete_map=color_map,
        title='Daily Sales Over Time',
        labels={'sale_date': 'Date', 'units_sold': 'Units Sold', 'product_name': 'Product'},
        # Markers adds dots to data points for improved readability
        markers=True,
        template=template
    )
    # Apply Theme
    fig_daily = apply_plotly_theme(fig_daily, template, theme_mode, height=400)
    st.plotly_chart(fig_daily, use_container_width=True)

    # ===== Total Sales by Product (Bar Chart) =====

    # Calculates total units sold per product
    product_totals = filtered_df.groupby('product_name')['units_sold'].sum().reset_index()
    # Creates a bar chart comparing sales
    fig_products = px.bar(
        product_totals,
        x='product_name',
        y='units_sold',
        color='product_name',
        color_discrete_map=color_map,
        title='Total Sales by Product',
        labels={'product_name': 'Product', 'units_sold': 'Units Sold'},
        template=template
    )
    # Removes legend because the bar labels show product names
    fig_products.update_layout(showlegend=False)
    fig_products = apply_plotly_theme(fig_products, template, theme_mode, height=400)
    st.plotly_chart(fig_products, use_container_width=True)

    # ===== Product Distribution Pie Chart =====
    fig_pie = px.pie(
        product_totals,
        values='units_sold',
        names='product_name',
        color='product_name',
        color_discrete_map=color_map,
        title='Sales Distribution',
        template=template
    )
    fig_pie = apply_plotly_theme(fig_pie, template, theme_mode, height=400)
    st.plotly_chart(fig_pie, use_container_width=True)

def prediction_dashboard():
    """
    Machine Learning Sales Prediction Dashboard

    Allows users to:
    - Generate sales forcasts
    - Compare model performance
    - Identify top-selling products
    - Export detailed prediction tables
    """

    # ===== Page Header =====
    st.title("🔮 Sales Prediction Dashboard")
    st.markdown("**ML-Powered Food Waste Minimization**")
    st.markdown("---")

    # ===== Load Data & Validation =====
    df = load_sales_data()
    # Data validation - Checks if data exists in the database
    if df is None or df.empty:
        st.warning("⚠️ No data available for predictions")
        return
    
    # ===== Initialise Session State =====
    # Session state stores results between user interactions
    # Holds prediction results and model evaluation metrics
    # so they persist when buttons are clicked or tabs switched
    if "predictions" not in st.session_state:
        st.session_state.predictions = {}
    if "prediction_scores" not in st.session_state:
        st.session_state.prediction_scores = {}
    if "prediction_algorithm" not in st.session_state:
        st.session_state.prediction_algorithm = "Random Forest"

    # ===== Theme & template =====
    theme_mode = st.session_state.get("theme_mode", "Light Mode")
    template = "plotly_white" if theme_mode == "Light Mode" else "plotly_dark"

    category_colors = {
        'Americano': "#FF5733",
        'Cappuccino': '#33C1FF',
        'Croissants': "#F460AA"
    }

    # Optional: assign distinct colors per product for consistency
    product_colors = {}
    all_products = sorted(df['product_name'].unique())
    base_colors = px.colors.qualitative.Safe  # bright, varied colors
    for i, prod in enumerate(all_products):
        product_colors[prod] = base_colors[i % len(base_colors)]

    tabs = st.tabs([
        "📈 Predictions",
        "🎯 Model Evaluation",
        "🏆 Top Products",
        "🔍 Detailed View"
    ])

    # =================== Tab 1: Predictions ===================
    # User can select products and a training period
    # generates machine learning forecasts for future sales
    with tabs[0]:
        st.header("Generate Sales Predictions")
        col1, col2 = st.columns([2, 1])

        # User selects products for prediction
        with col1:
            selected_products = st.multiselect(
                "Select products to predict:",
                options=all_products,
                default=all_products[:min(3, len(all_products))]
            )

        # User selects how many weeks of historicald data should be used
        # to train the model
        with col2:
            training_weeks = st.slider("Training period (weeks):", 4, 8, 4)

        if not selected_products:
            st.info("👆 Select at least one product")
            return
        
        # User selects which machine learning algorithm will be used
        algorithm_choice = st.radio(
            "Choose ML Algorithm:",
            ["Random Forest", "Gradient Boosting", "Linear Regression", "Support Vector Regression"],
            horizontal=True
        )

        # When user clicks the button, model is trained and prediction generated
        generate_button = st.button("🚀 Generate Predictions", type="primary")
        if generate_button:
            st.session_state.predictions = {}
            st.session_state.prediction_scores = {}

            progress_bar = st.progress(0)
            status_text = st.empty()

            # Model Training Loop
            # Train a seperate model for each selected product
            # Allows each product to have its own demand pattern
            for idx, product in enumerate(selected_products):
                status_text.text(f"Training model for {product}...")

                # Converts raw data into ready features
                X, y, feature_cols = prepare_ml_features(df, product, training_weeks)
                if X is None:
                    st.warning(f"⚠️ Not enough data for {product}")
                    continue

                # Train all ML models and store their performance metrics
                trained_models, model_scores = train_models(X, y)
                if algorithm_choice not in trained_models:
                    st.error(f"Could not train {algorithm_choice} for {product}")
                    continue

                model = trained_models[algorithm_choice]
                scores = model_scores[algorithm_choice]

                # Generate future sales predictions for the
                # next 28 days 
                product_data = df[df['product_name'] == product].sort_values('sale_date')
                predictions_df = predict_future_sales(model, product_data, feature_cols, days_ahead=28)

                # Save predictions and scores to session state
                # so that the results persist
                st.session_state.predictions[product] = predictions_df
                st.session_state.prediction_scores[product] = scores

                progress_bar.progress((idx + 1) / len(selected_products))

            status_text.text("✅ Complete!")
            progress_bar.empty()
            st.success(f"✅ Generated predictions for {len(st.session_state.predictions)} product(s)")
            st.markdown("---")

        # ----- Display each product prediction -----

        # Show predictions if they exist
        if st.session_state.predictions:

            prediction_window = st.selectbox(
                "Prediction View Window",
                [
                    "Next 3 days",
                    "Next 7 days",
                    "Next 14 days",
                    "Full 4 weeks"
                ]
            )

            window_days_map = {
                "Next 3 days": 3,
                "Next 7 days": 7,
                "Next 14 days": 14,
                "Full 4 weeks": 28
            }

            window_days = window_days_map[prediction_window]

            for product in st.session_state.predictions:
                with st.expander(f"📦 {product} - Next 4 Weeks", expanded=True):
                    pred_df = st.session_state.predictions[product]
                    pred_df = pred_df.head(window_days).copy()
                    scores = st.session_state.prediction_scores[product]

                    # ===== Metrics =====
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Avg Daily", f"{pred_df['predicted_sales'].mean():.1f}")
                    col2.metric("Total 4 Weeks", f"{pred_df['predicted_sales'].sum():.0f}")
                    col3.metric("Accuracy", f"{scores['Accuracy']:.1f}%")
                    col4.metric("RMSE", f"{scores['RMSE']:.2f}")

                    # ===== Historical + Predicted Sales Chart =====
                    historical = df[df['product_name'] == product].tail(28)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=historical['sale_date'],
                        y=historical['units_sold'],
                        name='Historical',
                        mode='lines+markers',
                        line=dict(color=product_colors[product], width=2)
                    ))
                    fig.add_trace(go.Scatter(
                        x=pred_df['date'],
                        y=pred_df['predicted_sales'],
                        name='Predicted',
                        mode='lines+markers',
                        line=dict(color=product_colors[product], width=2, dash='dash')
                    ))
                    fig.update_layout(title=f"{product} Sales Forecast")
                    fig = apply_plotly_theme(fig, template, theme_mode, height=400)
                    st.plotly_chart(fig, use_container_width=True)

                    # ===== Daily / Weekly / Production Tabs =====
                    tab1, tab2= st.tabs(["📅 Daily", "📊 Weekly"])
                    with tab1:
                        display_df = pred_df.copy()
                        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
                        display_df['day'] = pd.to_datetime(display_df['date']).dt.day_name()
                        table = display_df[['date', 'day', 'predicted_sales']].rename(
                            columns={'date': 'Date', 'day': 'Day', 'predicted_sales': 'Sales'}
                        )

                        st.markdown(table.to_html(index=False), unsafe_allow_html=True)

                    with tab2:
                        weekly_df = pred_df.copy()
                        weekly_df['week'] = pd.to_datetime(weekly_df['date']).dt.isocalendar().week

                        weekly = weekly_df.groupby('week')['predicted_sales'].agg(['sum', 'mean']).round(1).reset_index()
                        weekly.rename(columns={'week': 'Week', 'sum': 'Total Sales', 'mean': 'Avg Daily'}, inplace=True)
                        st.markdown(weekly.to_html(index=False), unsafe_allow_html=True)

    # =================== Tab 2: Model Evaluation ===================
    # Compare machine learning algorithms to determine
    # which model performs best for a selected product
    with tabs[1]:
        st.header("🎯 Model Evaluation")
        eval_product = st.selectbox("Product:", all_products)
        eval_weeks = st.slider("Training weeks:", 4, 8, 4, key="eval")

        # Trains all algorithms and compares them
        if st.button("📊 Evaluate All Models"):
            X, y, feature_cols = prepare_ml_features(df, eval_product, eval_weeks)
            if X is not None:
                trained_models, model_scores = train_models(X, y)
                scores_df = pd.DataFrame(model_scores).T.sort_values('Accuracy', ascending=False)

                col1, col2 = st.columns(2)
                with col1:
                    fig = px.bar(
                        scores_df.reset_index(),
                        x='index', y='Accuracy',
                        title='Model Accuracy',
                        color='Accuracy',
                        template=template,
                        color_continuous_scale=px.colors.sequential.Viridis
                    )
                    fig = apply_plotly_theme(fig, template, theme_mode)
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    fig = px.bar(
                        scores_df.reset_index(),
                        x='index', y='RMSE',
                        title='Model RMSE',
                        color='RMSE',
                        template=template,
                        color_continuous_scale=px.colors.sequential.Inferno
                    )
                    fig = apply_plotly_theme(fig, template, theme_mode)
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown(scores_df.to_html(), unsafe_allow_html=True)

        # Compare training period
        compare_button = st.button("Compare Training Periods (4-8 weeks)")
        
        if compare_button:

            comparison_results = []

            for weeks in range(4, 9):

                X, y, feature_cols = prepare_ml_features(df, eval_product, weeks)

                if X is None:
                    continue

                trained_models, model_scores = train_models(X, y)

                # Prevents crashing
                if not model_scores:
                    continue
                
                # Choose best model by RMSE
                best_model = min(model_scores.items(), key=lambda x: x[1]['RMSE'])

                model_name = best_model[0]
                scores = best_model[1]

                comparison_results.append({
                    "Training Weeks": weeks,
                    "Best Model": model_name,
                    "Accuracy": round(scores["Accuracy"], 2),
                    "RMSE": round(scores["RMSE"], 2)
                })

            if comparison_results:

                comp_df = pd.DataFrame(comparison_results)

                st.subheader("📊 Training Period Comparison")
                st.markdown(comp_df.to_html(index=False), unsafe_allow_html=True)
                
                best_row = comp_df.sort_values("RMSE").iloc[0]

                # ===== Display best training period =====
                st.success(
                    f"Best training period: **{best_row['Training Weeks']} weeks** "
                    f"using **{best_row['Best Model']}** "
                    f"(RMSE: {best_row['RMSE']})"
                # =========================================
)
                col1, col2 = st.columns(2)

                with col1:
                    fig = px.line(
                        comp_df,
                        x="Training Weeks",
                        y="Accuracy",
                        markers=True,
                        title="Accuracy vs Training Period",
                        template=template
                    )
                    fig = apply_plotly_theme(fig, template, theme_mode)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    fig = px.line(
                        comp_df,
                        x="Training Weeks",
                        y="RMSE",
                        markers=True,
                        title="RMSE vs Training Period",
                        template=template
                    )
                    fig = apply_plotly_theme(fig, template, theme_mode)
                    st.plotly_chart(fig, use_container_width=True)

    # =================== Tab 3: Top Products ===================
    # Identify the best selling products within a time period
    with tabs[2]:
        st.header("🏆 Top Products")
        # Allows user to analyse sales over the last
        # 4 weeks/8 weeks or entire period
        period = st.selectbox("Period:", ["Last 4 weeks", "Last 8 weeks", "All time"])
        if period == "Last 4 weeks":
            cutoff = df['sale_date'].max() - timedelta(weeks=4)
        elif period == "Last 8 weeks":
            cutoff = df['sale_date'].max() - timedelta(weeks=8)
        else:
            cutoff = df['sale_date'].min()
        period_df = df[df['sale_date'] >= cutoff]

        col1, col2 = st.columns(2)
        # ===== Top Coffee Products =====
        # Calculates and displays the top 3 coffee products
        with col1:
            st.subheader("☕ Top 3 Coffees")
            coffee = period_df[period_df['category'] == 'coffee']
            top_coffee = coffee.groupby('product_name')['units_sold'].sum().sort_values(ascending=False).head(3)
            for idx, (prod, sales) in enumerate(top_coffee.items(), 1):
                st.metric(f"#{idx} {prod}", f"{sales:,.0f}")
            if not top_coffee.empty:
                fig = px.line(
                    coffee[coffee['product_name'].isin(top_coffee.index)],
                    x='sale_date', y='units_sold',
                    color='product_name',
                    title='Sales Fluctuation',
                    markers=True,
                    color_discrete_map={p: product_colors[p] for p in top_coffee.index},
                    template=template
                )
                fig = apply_plotly_theme(fig, template, theme_mode)
                st.plotly_chart(fig, use_container_width=True)
        # ===== Top Croissant Products =====
        # Calculates and displays the top 3 croissant products
        with col2:
            st.subheader("🥐 Top 3 Croissants")
            croissant = period_df[period_df['category'] == 'croissants']
            if not croissant.empty:
                top_croissant = croissant.groupby('product_name')['units_sold'].sum().sort_values(ascending=False).head(3)
                for idx, (prod, sales) in enumerate(top_croissant.items(), 1):
                    st.metric(f"#{idx} {prod}", f"{sales:,.0f}")
                fig = px.line(
                    croissant[croissant['product_name'].isin(top_croissant.index)],
                    x='sale_date', y='units_sold',
                    color='product_name',
                    title='Sales Fluctuation',
                    markers=True,
                    color_discrete_map={p: product_colors[p] for p in top_croissant.index},
                    template=template
                )
                fig = apply_plotly_theme(fig, template, theme_mode)
                st.plotly_chart(fig, use_container_width=True)

    # =================== Tab 4: Detailed View ===================
    # Generates a detailed prediction table for a specific product
    # Allows user to download result as a CSV file
    with tabs[3]:
        st.header("🔍 Detailed View")
        detail_product = st.selectbox("Product:", all_products, key="detail")
        col1, col2 = st.columns(2)
        with col1:
            detail_weeks = st.slider("Training weeks:", 4, 8, 4, key="detail_weeks")
        with col2:
            detail_algo = st.selectbox("Algorithm:",
                ["Random Forest", "Gradient Boosting", "Linear Regression", "Support Vector Regression"])
        if st.button("🔮 Generate"):
            X, y, feature_cols = prepare_ml_features(df, detail_product, detail_weeks)
            if X is not None:
                trained_models, _ = train_models(X, y)
                if detail_algo in trained_models:
                    model = trained_models[detail_algo]
                    product_data = df[df['product_name'] == detail_product].sort_values('sale_date')
                    predictions_df = predict_future_sales(model, product_data, feature_cols, 28)

                    # Date formatting
                    predictions_df['date_str'] = predictions_df['date'].dt.strftime('%Y-%m-%d')
                    predictions_df['day'] = predictions_df['date'].dt.day_name()
                    predictions_df['week'] = predictions_df['date'].dt.isocalendar().week

                    display = predictions_df[['date_str', 'day', 'predicted_sales', 'week']]
                    display.columns = ['Date', 'Day', 'Sales', 'Week']

                    st.markdown(display.to_html(index=False), unsafe_allow_html=True)

                    # Allow user to download prediction results
                    csv = display.to_csv(index=False)
                    st.download_button(
                        "📥 Download CSV",
                        csv,
                        f"predictions_{detail_product}_{datetime.now().strftime('%Y%m%d')}.csv",
                        "text/csv"
                    )

# ============ MAIN APP ============

def main():

    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False

    # ===== Accessibility & Theme Controls =====
    st.sidebar.markdown("---")
    st.sidebar.header("♿ Accessibility")

    font_size = st.sidebar.slider(
        "Font Size",
        min_value=14,
        max_value=26,
        value=16
    )

    # ===== Theme Toggle =====
    #if "theme_mode" not in st.session_state:
        #st.session_state.theme_mode = "Dark Mode"  # default dark

    #with st.sidebar:
        #st.markdown("## 🎨 Theme")
        #theme_choice = st.radio(
            #"Select theme:",
            #["Light Mode", "Dark Mode"],
            #index=0 if st.session_state.theme_mode == "Light Mode" else 1
        #)
        #st.session_state.theme_mode = theme_choice

    # Apply global CSS to Streamlit components
    #apply_global_theme(st.session_state.theme_mode)
    #apply_streamlit_theme(st.session_state.theme_mode)

    # Apply font size globally
    st.markdown(
        f"""
        <style>
        html, body, [class*="css"]  {{
            font-size: {font_size}px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # ===== Call dashboard pages =====
    if not st.session_state.data_loaded:
        upload_page()
    else:
        # Mode selector in sidebar
        st.sidebar.title("🎛️ Dashboard Mode")
        mode = st.sidebar.radio("Select Mode:", ["📊 Analysis", "🔮 Predictions"], label_visibility="collapsed")
        st.sidebar.markdown("---")
        if st.sidebar.button("📤 Upload New Data", use_container_width=True):
            st.session_state.data_loaded = False
            st.rerun()
        if mode == "📊 Analysis":
            analysis_dashboard()
        else:
            prediction_dashboard()

if __name__ == "__main__":
    main()