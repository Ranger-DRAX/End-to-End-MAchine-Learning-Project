# 🎓 Student Performance Prediction - End-to-End ML Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-green)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

An end-to-end machine learning project that predicts student math scores based on various demographic and academic factors using multiple regression algorithms with hyperparameter tuning.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Technical Architecture](#technical-architecture)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Models Evaluated](#models-evaluated)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Endpoints](#api-endpoints)
- [Results & Outcomes](#results--outcomes)
- [Deployment](#deployment)
- [Technologies Used](#technologies-used)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Project Overview

This project implements a complete machine learning pipeline to predict student math performance based on multiple factors including:
- Gender
- Race/Ethnicity
- Parental level of education
- Lunch type
- Test preparation course completion
- Reading score
- Writing score

The system uses a modular architecture with separate components for data ingestion, transformation, model training, and prediction, all wrapped in a Flask web application.

## ❓ Problem Statement

Educational institutions need to identify students who may need additional support to improve their academic performance. This project builds a predictive model to forecast student math scores based on demographic and academic factors, enabling early intervention and personalized support.

## 📊 Dataset

The project uses a student performance dataset containing:
- **Source**: Student academic performance data
- **Location**: `notebook/data/studentData.csv`
- **Target Variable**: `math_score` (student's mathematics test score)
- **Features**:
  - **Categorical**: gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course
  - **Numerical**: reading_score, writing_score

## 🏗️ Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Flask Web Application                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Prediction Pipeline                       │
│  ┌────────────────┐        ┌──────────────────┐            │
│  │ Data Ingestion │───────▶│ Data Transform   │            │
│  └────────────────┘        └──────────────────┘            │
│                                     │                        │
│                                     ▼                        │
│                            ┌──────────────────┐             │
│                            │  Model Training  │             │
│                            └──────────────────┘             │
│                                     │                        │
│                                     ▼                        │
│                            ┌──────────────────┐             │
│                            │   Best Model     │             │
│                            │   (Artifacts)    │             │
│                            └──────────────────┘             │
└─────────────────────────────────────────────────────────────┘
```

## 🔄 Machine Learning Pipeline

### 1. **Data Ingestion Component** (`src/components/data_ingestion.py`)
- Reads raw student data from CSV
- Splits data into training (80%) and testing (20%) sets
- Saves raw, train, and test datasets to `artifacts/` directory
- Implements logging for tracking pipeline execution

### 2. **Data Transformation Component** (`src/components/data_transformation.py`)
- **Numerical Features Pipeline**:
  - Median imputation for missing values
  - Standard scaling for normalization
  
- **Categorical Features Pipeline**:
  - Most frequent imputation for missing values
  - One-Hot Encoding for categorical variables
  - Standard scaling (with_mean=False for sparse matrices)

- Saves preprocessing object as pickle file for inference

### 3. **Model Training Component** (`src/components/model_trainer.py`)
- Evaluates multiple regression algorithms
- Performs hyperparameter tuning using GridSearchCV
- Selects best performing model based on R² score
- Saves trained model as pickle file

### 4. **Prediction Pipeline** (`src/pipeline/predict_pipeline.py`)
- Loads trained model and preprocessor
- Transforms input data using saved preprocessor
- Generates predictions for new student data

## 🤖 Models Evaluated

The project evaluates 7 different regression algorithms with hyperparameter tuning:

| Model | Key Hyperparameters Tuned |
|-------|---------------------------|
| **Linear Regression** | Default configuration |
| **Decision Tree** | criterion (squared_error, friedman_mse, absolute_error, poisson) |
| **Random Forest** | n_estimators (8, 16, 32, 64, 128, 256) |
| **Gradient Boosting** | learning_rate, subsample, n_estimators |
| **XGBoost Regressor** | learning_rate, n_estimators |
| **CatBoost Regressor** | depth, learning_rate, iterations |
| **AdaBoost Regressor** | learning_rate, n_estimators |

**Model Selection Criteria**: R² Score (Coefficient of Determination)

## 💻 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/Ranger-DRAX/End-to-End-MAchine-Learning-Project.git
cd End-to-End-MAchine-Learning-Project
```

2. **Create and activate virtual environment**

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

This will install:
- pandas, numpy (Data manipulation)
- scikit-learn (ML algorithms)
- catboost, xgboost (Gradient boosting)
- Flask (Web framework)
- gunicorn (Production server)
- seaborn, matplotlib (Visualization)

4. **Install project as package**
```bash
pip install -e .
```

## 🚀 Usage

### Training the Model

```bash
python src/pipeline/train_pipeline.py
```

This will:
1. Ingest data from `notebook/data/studentData.csv`
2. Perform train-test split
3. Apply data transformations
4. Train and evaluate all models
5. Save the best model to `artifacts/model.pkl`
6. Save the preprocessor to `artifacts/preprocessor.pkl`

### Running the Web Application

```bash
python app.py
```

The application will start on `http://0.0.0.0:5000`

Access the web interface at `http://localhost:5000`

### Making Predictions

**Via Web Interface:**
1. Navigate to `http://localhost:5000`
2. Click on prediction form
3. Fill in student information:
   - Gender
   - Race/Ethnicity
   - Parental Level of Education
   - Lunch Type
   - Test Preparation Course
   - Reading Score
   - Writing Score
4. Submit to get predicted math score

**Via Python Code:**
```python
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

# Create custom data
data = CustomData(
    gender="female",
    race_ethnicity="group B",
    parental_level_of_education="bachelor's degree",
    lunch="standard",
    test_preparation_course="completed",
    reading_score=75,
    writing_score=78
)

# Convert to DataFrame
df = data.get_data_as_data_frame()

# Make prediction
pipeline = PredictPipeline()
prediction = pipeline.predict(df)
print(f"Predicted Math Score: {prediction[0]}")
```

## 📁 Project Structure

```
End-to-End-MAchine-Learning-Project/
│
├── artifacts/                      # Saved models and preprocessors
│   ├── model.pkl                   # Trained ML model
│   ├── preprocessor.pkl            # Data transformation pipeline
│   ├── train.csv                   # Training dataset
│   ├── test.csv                    # Testing dataset
│   └── data.csv                    # Raw dataset
│
├── notebook/                       # Jupyter notebooks for EDA
│   └── data/
│       └── studentData.csv         # Original dataset
│
├── src/                            # Source code
│   ├── __init__.py
│   ├── exception.py                # Custom exception handling
│   ├── logger.py                   # Logging configuration
│   ├── utils.py                    # Utility functions
│   │
│   ├── components/                 # ML pipeline components
│   │   ├── __init__.py
│   │   ├── data_ingestion.py      # Data loading & splitting
│   │   ├── data_transformation.py # Feature engineering
│   │   └── model_trainer.py       # Model training & evaluation
│   │
│   └── pipeline/                   # Training & prediction pipelines
│       ├── __init__.py
│       ├── train_pipeline.py      # End-to-end training
│       └── predict_pipeline.py    # Inference pipeline
│
├── templates/                      # HTML templates
│   ├── index.html                 # Landing page
│   └── home.html                  # Prediction form
│
├── app.py                         # Flask application
├── setup.py                       # Package configuration
├── requirements.txt               # Python dependencies
├── Procfile                       # Deployment configuration
├── .gitignore                     # Git ignore rules
└── README.md                      # Project documentation
```

## 🌐 API Endpoints

### `GET /`
- **Description**: Landing page
- **Returns**: HTML page with project information

### `GET /predictdata`
- **Description**: Displays prediction form
- **Returns**: HTML form for data input

### `POST /predictdata`
- **Description**: Accepts form data and returns prediction
- **Request Body**: Form data with student information
- **Returns**: HTML page with predicted math score

## 📈 Results & Outcomes

### Model Performance
The project successfully:
- ✅ Implements modular ML pipeline with clear separation of concerns
- ✅ Evaluates 7 different regression algorithms
- ✅ Performs automated hyperparameter tuning using GridSearchCV
- ✅ Selects best model based on R² score on test data
- ✅ Achieves production-ready performance with proper error handling

### Key Features Implemented
1. **Robust Pipeline Architecture**: Modular components for easy maintenance
2. **Automated Model Selection**: Evaluates multiple algorithms automatically
3. **Hyperparameter Optimization**: Grid search for best parameters
4. **Exception Handling**: Custom exception class for debugging
5. **Logging**: Comprehensive logging for monitoring
6. **Web Interface**: User-friendly Flask application
7. **Scalability**: Package structure for easy deployment

### Business Impact
- 📊 Enables early identification of at-risk students
- 🎯 Supports data-driven educational interventions
- ⚡ Provides instant predictions through web interface
- 📈 Scalable architecture for handling multiple predictions

## 🚢 Deployment

### Local Development
```bash
python app.py
```

### Production Deployment (with Gunicorn)
```bash
gunicorn app:app
```

### Environment Configuration
The `Procfile` is configured for platforms like Heroku:
```
web: gunicorn app:app
```

### Deployment Checklist
- [ ] Set `debug=False` in production
- [ ] Configure environment variables
- [ ] Set up proper logging
- [ ] Configure CORS if needed
- [ ] Set up database for storing predictions (optional)
- [ ] Implement authentication (if required)

## 🛠️ Technologies Used

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.8+ |
| **Web Framework** | Flask |
| **ML Libraries** | scikit-learn, XGBoost, CatBoost |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Server** | Gunicorn |
| **Version Control** | Git, GitHub |

## 🔮 Future Enhancements

- [ ] Add database integration for storing predictions
- [ ] Implement user authentication and authorization
- [ ] Create REST API with JSON responses
- [ ] Add model versioning and A/B testing
- [ ] Implement CI/CD pipeline
- [ ] Add unit tests and integration tests
- [ ] Create Docker containerization
- [ ] Add real-time model monitoring
- [ ] Implement batch prediction capabilities
- [ ] Add model explainability (SHAP values)
- [ ] Create comprehensive dashboards

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Arafat**
- GitHub: [@Ranger-DRAX](https://github.com/Ranger-DRAX)

## 🙏 Acknowledgments

- Dataset providers
- Open-source community
- scikit-learn, XGBoost, and CatBoost teams

---

⭐ **If you find this project helpful, please consider giving it a star!** ⭐