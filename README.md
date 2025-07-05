# Week 1: Linear Regression - Housing Price Prediction

## Project Overview
This project implements a linear regression model to predict housing prices based on various features. It's part of a weekly AI project series where we build different machine learning models from scratch.

## What You'll Learn
- Data preprocessing and exploration
- Feature engineering
- Linear regression implementation
- Model evaluation and validation
- Data visualization
- Cross-validation techniques

## Project Structure
```
week-01-linear-regression/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── housing_prediction.ipynb  # Jupyter notebook with complete analysis
├── housing_prediction.py     # Python script version
├── data/                     # Data directory
│   └── housing_data.csv      # Sample housing dataset
└── models/                   # Saved models
    └── linear_regression_model.pkl
```

## Features Used for Prediction
- Square footage of the house
- Number of bedrooms
- Number of bathrooms
- Age of the house
- Distance to city center
- Crime rate in the area
- School rating
- And more...

## Getting Started

### Prerequisites
- Python 3.7+
- Jupyter Notebook (optional, for interactive analysis)

### Installation
1. Clone or download this project
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project
1. **Jupyter Notebook** (Recommended for learning):
   ```bash
   jupyter notebook housing_prediction.ipynb
   ```

2. **Python Script**:
   ```bash
   python housing_prediction.py
   ```

## Model Performance
The linear regression model typically achieves:
- R² Score: ~0.75-0.85
- Mean Absolute Error: ~$25,000-35,000
- Root Mean Square Error: ~$40,000-50,000

## Key Concepts Covered
1. **Data Exploration**: Understanding the dataset structure and relationships
2. **Feature Engineering**: Creating new features and handling missing data
3. **Model Training**: Implementing linear regression with scikit-learn
4. **Evaluation**: Using multiple metrics to assess model performance
5. **Visualization**: Creating plots to understand data and model behavior
6. **Cross-validation**: Ensuring model robustness

## Next Steps
After completing this project, you'll be ready for:
- Week 2: Logistic Regression (Classification)
- Week 3: Decision Trees
- Week 4: Random Forests
- And more advanced algorithms!

## Contributing
Feel free to experiment with:
- Different feature combinations
- Advanced preprocessing techniques
- Ensemble methods
- Hyperparameter tuning

Happy coding! 🏠📊 