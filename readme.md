# Chronic Disease Prediction using Word2Vec and Similarity Scores

This project aims to predict chronic diseases based on patient electronic health records (EHR) using Word2Vec embeddings and similarity scores.

## Project Overview

- **Data Loading and Preprocessing**: The patient EHR dataset is loaded from a CSV file. Date columns are converted to datetime objects, irrelevant columns are dropped, and relevant columns are selected.

- **Word Embedding**: Categorical variables in the dataset are embedded using the Word2Vec model trained on the training dataset. This allows us to represent categorical variables as continuous vectors.

- **Model Training**: For each chronic disease category, the dataset is filtered, and combined vectors of selected features and their embeddings are created.

- **Prediction**: Given a new set of input features, the model calculates the similarity scores between the input and each combined vector for all chronic diseases. The chronic disease with the highest similarity score is predicted as the output.

- **Manual and Automated Testing**: Users can test the model with a sample input from the test dataset or manually input their own test data.

- **Accuracy Evaluation**: The accuracy of the model is calculated by comparing predictions on the test dataset with actual labels.

## Running the Project

### Prerequisites
- Python 3.x
- Pandas
- Gensim
- scikit-learn

### Installation
1. Clone this repository:

    ```bash
    git clone git@github.com:Anurag9689/chronic-disease-prediction.git
    ```

2. Navigate into 'chronic-disease-prediction' project `cd chronic-disease-prediction/`

#### Python Environment Setup

3. **Create Python 3 Virtual Environment**:
   - For Linux/macOS:

     ```bash
     python3 -m venv env
     ```

   - For Windows:

     ```bash
     python -m venv env
     ```

4. **Activate the Environment**:
   - For Linux/macOS:

     ```bash
     source env/bin/activate
     ```

   - For Windows:

     ```bash
     .\env\Scripts\activate
     ```

5. **Install Dependencies**:
   - After activating the environment, install the required dependencies using pip:

     ```bash
     pip install -r requirements.txt
     ```

### Usage
1. **Data Preparation**:
    - Place your patient EHR dataset in CSV format in the project directory.

2. **Run the Code**:
    - Open a terminal and navigate to the project directory.
    - Run the main script:

    ```bash
    python3 chronic_disease_prediction.py
    ```

3. **Testing**:
    - Follow the prompts to choose between manual or automated testing.
    - Provide input data as requested.
    - View the predicted chronic disease.

4. **Accuracy Evaluation**:
    - Choose to evaluate the accuracy on the test dataset.
    - View the accuracy of the model.

## Sample Input

Here's an example of a sample input:

```python
{
    'age': 45,
    'blood_glucose_levels': 120,
    'cholesterol_levels': 200,
    'heart_rate': 70,
    'gender': 'Male',
    'education level': 'Graduate',
    'smoking_status': 'Non-smoker',
    'physical activity level': 'Active',
    'dietary habits': 'Balanced diet',
    'alcohol consumption': 'Occasional drinker',
    'health status': 'Good',
    'family_history': 'None',
    'air_pollution_levels': 'Low',
    'geographic_location': 'Urban',
    'climate_conditions': 'Mild',
    'mental_health_status': 'Stable',
    'stress_levels': 'Low',
    'coping_mechanisms': 'Exercise',
    'adherence_to_treatment': 'Regular',
    'healthcare_utilization_patterns': 'Routine check-ups',
    'medication': 'None'
}
```