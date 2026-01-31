# 🎮 Global Video Game Sales Analytics

An interactive web application for analyzing and predicting video game sales data for PS4 and Xbox One platforms. This project provides comprehensive analytics, visualizations, and machine learning-based sales predictions using a Random Forest model.

## 📋 Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Sources](#data-sources)
- [Dashboard Features](#dashboard-features)
- [Machine Learning Model](#machine-learning-model)
- [Future Enhancements](#future-enhancements)

## ✨ Features

- **📊 Real-time KPI Dashboard** - Total games, global sales, and statistical metrics at a glance
- **📈 Interactive Visualizations** - Plotly-powered charts with hover tooltips:
  - Platform-wise sales comparison (PS4 vs Xbox One)
  - Genre-wise sales analysis ranked by performance
  - Year-wise sales trends showing market evolution
  - Regional sales distribution across markets
- **🔮 ML-powered Sales Prediction** - Random Forest model predicting sales from platform and genre
- **🌍 Regional Market Breakdown** - Sales across North America, Europe, Japan, Rest of World
- **👑 Publisher Rankings** - Top 10 publishers by global sales
- **🎮 Game Benchmarking** - Top 15 best-selling games with detailed metrics
- **⚡ Optimized Performance** - Data and model caching for instant insights
- **📋 Dataset Explorer** - Expandable section with comprehensive data statistics

## 🛠 Technologies Used

- **Python 3.7+** - Programming language
- **Streamlit** - Interactive web application framework
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing and array operations
- **Plotly** - Interactive and responsive visualizations
- **Scikit-learn** - Machine learning models and preprocessing
  - RandomForestRegressor - Sales prediction
  - OneHotEncoder - Categorical feature encoding
  - ColumnTransformer - Pipeline preprocessing

## 📁 Project Structure

```
Global-Video-Game-Sales-Analytics/
│
├── streamlit_app.py              # Main Streamlit web application with dashboard & ML model
├── Game.csv                      # Combined video game sales dataset (cleaned)
├── game_sales_analysis.sql       # SQL queries for data analysis and insights
├── requirements.txt              # Python package dependencies
└── README.md                      # Project documentation
```

### Key Files

| File | Purpose |
|------|---------|
| `streamlit_app.py` | Main application with data pipeline, visualizations, and ML prediction model |
| `Game.csv` | Cleaned dataset containing ~1,500-2,000 games with sales data |
| `game_sales_analysis.sql` | SQL queries for backend data analysis, aggregations, and exploratory queries |
| `requirements.txt` | Streamlit, Pandas, NumPy, Scikit-learn, Plotly dependencies |

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/sathish-2424/Global-Video-Game-Sales-Analytics
   cd Global-Video-Game-Sales-Analytics
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   The required packages include:
   - `streamlit` - Web application framework
   - `pandas` - Data manipulation and analysis
   - `numpy` - Numerical computing
   - `scikit-learn` - Machine learning models
   - `plotly` - Interactive visualization library

## 💻 Usage

### Running the Streamlit Application

1. **Start the application**
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Access the dashboard**
   - The app automatically opens at `http://localhost:8501`
   - Runs on your local machine with hot-reload enabled

3. **Dashboard Sections**
   - **Key Business Metrics** - Total games, global sales, averages
   - **Sales Insights** - Interactive charts for platform, genre, and year-wise analysis
   - **Regional Distribution** - Sales breakdown by North America, Europe, Japan, Rest of World
   - **Publisher & Game Analysis** - Top publishers and best-selling games
   - **Sales Prediction** - ML-powered forecasting tool

### Making Predictions

1. Navigate to the **Sales Prediction Simulator** section
2. Select a **Platform** (PS4 or Xbox One)
3. Select a **Genre** (Action, Adventure, RPG, Sports, etc.)
4. Click **"🔮 Predict Sales"**
5. View predicted global sales in million units

## 📊 Data Overview

### Dataset Statistics
- **Total Games**: ~1,500-2,000 games after data cleaning and quality filtering
- **Time Period**: 2013 - 2017 (5 years of sales data)
- **Platforms**: PS4 and Xbox One (primary focus)
- **Genres**: 12+ genres including Action, Adventure, RPG, Sports, Shooter, Strategy, Racing, Puzzle, and more
- **Publishers**: 500+ publishers including EA Sports, 2K Games, Ubisoft, Sony, Microsoft, and indie developers
- **Coverage**: Global sales tracked across 4 regions - North America, Europe, Japan, Rest of World
- **Data Quality**: 99%+ consistency with automated validation and cleaning

### Data Cleaning Pipeline

The application implements robust data preprocessing:

1. **Zero-Sales Removal** - Filters out games with zero global sales (~31% of raw data)
2. **Sales Reconciliation** - Recalculates global sales as sum of regional sales
3. **Data Type Optimization** - Converts categorical columns to category dtype for memory efficiency
4. **Year Standardization** - Ensures Year field is properly formatted as integer
5. **Publisher Name Normalization** - Standardizes publisher names across dataset

### Data Fields

| Field | Type | Description |
|-------|------|-------------|
| **Game** | String | Name of the video game |
| **Year** | Integer | Release year |
| **Genre** | Category | Game genre (Action, Adventure, RPG, etc.) |
| **Publisher** | String | Game publisher/developer |
| **Platform** | Category | Console platform (PS4, Xbox One) |
| **North America** | Float | Sales in North America (million units) |
| **Europe** | Float | Sales in Europe (million units) |
| **Japan** | Float | Sales in Japan (million units) |
| **Rest of World** | Float | Sales in other regions (million units) |
| **Global** | Float | Total global sales (million units) |

## 🎯 Dashboard Features

### 1. Key Business Metrics (Real-time)
- **🎯 Total Games** - Count of games in dataset
- **💰 Total Global Sales** - Sum of all global sales (millions of units)
- **📊 Average Sales** - Mean sales per game
- **📈 Median Sales** - Median sales per game (robust to outliers)

### 2. Interactive Visualizations (Plotly-powered)

#### Platform-wise Sales
- Compares global sales across gaming platforms
- Bar chart with exact values displayed
- Identify top-performing platforms

#### Genre-wise Sales
- Horizontal bar chart ranked by sales
- Highlights which genres drive revenue
- Sorted in descending order

#### Year-wise Trends
- Line chart with markers showing sales over time
- Interactive hover for precise year data
- Identifies growth periods and market cycles

#### Regional Distribution
- Sales breakdown by region: North America, Europe, Japan, Rest of World
- Visual representation of market concentration
- Quick identification of regional strengths

### 3. Advanced Analytics

#### Publisher Analysis
- Top 10 publishers by global sales
- Horizontal bar chart for easy comparison
- Identifies market leaders

#### Best-Selling Games
- Top 15 games with sales, platform, genre, and year
- Interactive sortable data table
- Benchmark performance comparisons

### 4. Sales Prediction Simulator
- **Interactive Form**: Select platform and genre dynamically
- **ML Model**: Trained Random Forest Regressor with 100 estimators
- **Output**: Predicted global sales in millions with confidence level
- **Confidence Indicator**: High/Medium/Low based on model R² score performance
- **Real-time Inference**: Instant predictions without page reload using cached model

### 5. Dataset Information
- Expandable section with dataset statistics
- Record count, time period, genre/publisher counts
- Data quality metrics
- Platform statistics

## 🤖 Machine Learning Model

### Model Architecture

| Component | Specification |
|-----------|---------------|
| **Algorithm** | Random Forest Regressor |
| **Number of Trees** | 100 estimators |
| **Max Depth** | 15 levels |
| **Min Samples Split** | 5 samples |
| **Feature Encoding** | One-Hot Encoding (categorical → numerical) |
| **Training Split** | 80% training, 20% validation |
| **Parallelization** | Multi-core processing (n_jobs=-1) |
| **Random Seed** | 42 (reproducible results) |

### Model Workflow

1. **Data Preparation**
   - Input features: Platform (categorical), Genre (categorical)
   - Target variable: Global sales (continuous)
   - Outlier removal using IQR method (Q1 - 1.5×IQR to Q3 + 1.5×IQR)

2. **Feature Engineering**
   - OneHotEncoder converts categorical features to binary vectors
   - ColumnTransformer applies preprocessing in pipeline

3. **Model Training**
   - Random Forest learns non-linear relationships
   - Training on filtered dataset (without extreme outliers)
   - Optimized for generalization on new platform-genre combinations

4. **Performance Metrics**
   - Training R² score - Model fit on training data
   - Test R² score - Generalization performance on unseen data
   - Displayed in prediction interface

5. **Inference & Caching**
   - Model cached using `@st.cache_resource` for performance
   - Zero cold-start latency after first training
   - Real-time predictions returned instantly

### Model Advantages
✅ Handles non-linear relationships between features  
✅ No need for extensive feature scaling  
✅ Robust to outliers in training  
✅ Fast inference time for predictions  
✅ Captures complex platform-genre interactions

## � Performance Optimizations

The application implements multiple optimization techniques:

| Optimization | Benefit |
|--------------|---------|
| **Data Caching** | `@st.cache_data` - Aggregations computed once, reused across sessions |
| **Model Caching** | `@st.cache_resource` - ML model trained once, instant predictions |
| **Categorical Types** | Memory-efficient storage for Platform and Genre columns |
| **Outlier Filtering** | Removes extreme values for better model generalization |
| **Parallel Processing** | Multi-core Random Forest (n_jobs=-1) for faster training |
| **Responsive Layout** | Two-column layout adapts to screen size |

## 🔧 Troubleshooting

### Issue: "File not found" error
**Solution**: Ensure `Game.csv` is in the same directory as `streamlit_app.py`

### Issue: Slow predictions on first run
**Solution**: Normal behavior - Streamlit caches the model after first training. Subsequent predictions are instant.

### Issue: Port 8501 already in use
**Solution**: Run the app on a different port:
```bash
streamlit run streamlit_app.py --server.port 8502
```

### Issue: Missing dependencies
**Solution**: Install all required packages:
```bash
pip install -r requirements.txt
```

## 📈 Future Enhancements

Potential improvements for future versions:

- **Feature Engineering**: Add developer sentiment, marketing spend, franchise history, rating scores
- **Model Improvements**: Try XGBoost, Neural Networks, or ensemble voting methods
- **Regional Predictions**: Separate prediction models for each geographic region
- **Time Series Analysis**: Temporal patterns, seasonal trends, and trend forecasting
- **Data Export**: Download reports and predictions as CSV/PDF with metadata
- **Comparative Analysis**: Compare predicted vs actual sales for model validation and accuracy tracking
- **User Feedback**: Rating system to continuously improve model predictions
- **Advanced Filters**: Dashboard filters by year range, publisher, rating threshold

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the MIT License - see the LICENSE file for details.

## 📧 Contact & Support

For issues, questions, or feature requests:
- Open an issue in the repository
- Check existing issues and discussions for solutions
- Review the Troubleshooting section in this README

---

**Enjoy analyzing video game sales data! 🎮📊**
