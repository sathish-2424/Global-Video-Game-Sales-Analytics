# 🎮 Video Game Sales Analytics

An interactive web application for analyzing and predicting video game sales data for PS4 and Xbox One platforms. This project provides comprehensive analytics, visualizations, and machine learning-based sales predictions.

## 📋 Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Sources](#data-sources)
- [Features Overview](#features-overview)
- [Machine Learning Model](#machine-learning-model)

## ✨ Features

- **📊 Key Performance Metrics**: Total games, total global sales, and average sales per game
- **📈 Interactive Visualizations**: 
  - Platform-wise sales comparison (PS4 vs Xbox One)
  - Genre-wise sales analysis
  - Year-wise sales trends
  - Regional sales distribution (PS4 by country)
- **🔮 Sales Prediction**: Machine learning model to predict global sales based on platform and genre
- **🌍 Regional Analysis**: Detailed breakdown of sales by region (North America, Europe, Japan, Rest of World)
- **📱 Interactive Dashboard**: User-friendly Streamlit interface with real-time analytics

## 🛠 Technologies Used

- **Python 3.x**
- **Streamlit** - Web application framework
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Plotly** - Interactive visualizations
- **Scikit-learn** - Machine learning (Random Forest Regressor)
- **Jupyter Notebook** - Data cleaning and preprocessing
- **Power BI** - Additional analytics and reporting

## 📁 Project Structure

```
Video-Game-Sales-Analytics/
│
├── streamlit_app.py          # Main Streamlit application
├── Data_Clean.ipynb          # Data cleaning and preprocessing notebook
├── requirements.txt          # Python dependencies
├── VG_Sales_Analysis.pbix    # Power BI analysis file
│
├── Data Files:
│   ├── PS4_GamesSales.csv    # Raw PS4 sales data
│   ├── XboxOne_GameSales.csv # Raw Xbox One sales data
│   ├── PS4.csv               # Cleaned PS4 data
│   └── XboxOne.csv           # Cleaned Xbox One data
│
└── README.md                 # Project documentation
```

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Video-Game-Sales-Analytics.git
   cd Video-Game-Sales-Analytics
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

## 💻 Usage

### Running the Streamlit App

1. **Start the application**
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Access the dashboard**
   - The app will automatically open in your default web browser
   - If not, navigate to `http://localhost:8501`

### Using the Data Cleaning Notebook

1. **Open Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Run the Data_Clean.ipynb notebook**
   - This notebook processes the raw CSV files
   - Handles missing values, duplicates, and data standardization
   - Generates cleaned CSV files (`PS4.csv` and `XboxOne.csv`)

### Features in the Dashboard

1. **Key Metrics Section**
   - View total number of games analyzed
   - See total global sales in millions
   - Check average sales per game

2. **Sales Insights**
   - Compare sales between PS4 and Xbox One platforms
   - Analyze which genres perform best
   - Track sales trends over the years

3. **Regional Analysis**
   - Explore PS4 sales distribution across different regions
   - Identify top-performing markets

4. **Sales Prediction**
   - Select a platform (PS4 or Xbox One)
   - Choose a game genre
   - Get predicted global sales in million units

## 📊 Data Sources

The project uses video game sales data including:
- **PS4 Games**: 1,033 games after cleaning
- **Xbox One Games**: 613 games after cleaning

### Data Fields

- **Game**: Name of the video game
- **Year**: Release year
- **Genre**: Game genre (Action, Adventure, RPG, etc.)
- **Publisher**: Game publisher
- **North America**: Sales in North America (million units)
- **Europe**: Sales in Europe (million units)
- **Japan**: Sales in Japan (million units)
- **Rest of World**: Sales in other regions (million units)
- **Global**: Total global sales (million units)

## 🎯 Features Overview

### 1. Key Business Metrics
- Real-time calculation of total games, sales, and averages
- Displayed in an easy-to-read metric format

### 2. Interactive Charts
- **Bar Charts**: Platform and genre comparisons
- **Line Charts**: Temporal sales trends
- **Horizontal Bar Charts**: Genre performance rankings
- All charts are interactive with Plotly

### 3. Regional Sales Analysis
- Detailed breakdown of PS4 sales by region
- Visual representation of market distribution
- Color-coded charts for easy interpretation

### 4. Sales Prediction Model
- **Algorithm**: Random Forest Regressor
- **Features**: Platform and Genre
- **Training**: 70% of data used for training
- **Output**: Predicted global sales in million units

## 🤖 Machine Learning Model

The project uses a **Random Forest Regressor** to predict game sales:

- **Preprocessing**: One-hot encoding for categorical features (Platform, Genre)
- **Model Parameters**:
  - 300 estimators
  - Random state: 42
  - Parallel processing enabled
- **Input Features**: Platform, Genre
- **Target Variable**: Global Sales (million units)

### Model Training
The model is trained on 70% of the combined dataset and cached for performance using Streamlit's caching mechanism.

## 📝 Notes

- The data cleaning notebook should be run first to generate the cleaned CSV files
- Missing values in Year and Publisher columns are handled during preprocessing
- Duplicate entries are removed based on Game and Year combination
- The Streamlit app expects `PS4.csv` and `XboxOne.csv` to be in the root directory

## 🔮 Future Enhancements

Potential improvements for the project:
- Add more platforms (Nintendo Switch, PC, etc.)
- Include additional features for prediction (Publisher, Year, etc.)
- Implement model evaluation metrics (R², MAE, RMSE)
- Add data export functionality
- Create time series forecasting
- Add user authentication for personalized dashboards

## 📄 License

This project is open source and available for educational and research purposes.

## 👤 Author

Created as a data analytics and machine learning project for video game sales analysis.

---

**Enjoy exploring video game sales data! 🎮📊**
