:

🏦 Loan Approval Prediction (ML + Streamlit)
📌 Abstract

Loan approval is a critical decision-making process in the banking sector. Traditional manual verification is time-consuming and prone to human bias.
This project leverages Machine Learning (Random Forest Classifier) to predict whether a loan should be approved or rejected based on applicant details such as income, education, dependents, property area, and credit history.
The trained model is deployed as a Streamlit Web Application for easy interaction.

🎯 Objectives

To build a classification ML model for predicting loan approval.

To preprocess both categorical and numerical features effectively.

To evaluate the model using accuracy and cross-validation.

To deploy the model into a web-based interface for real-time predictions.

🛠️ Methodology

Dataset Preparation

Collected a dataset of loan applicants (sample data).

Features included: Gender, Married Status, Dependents, Education, Employment, Income, Loan Amount, Loan Term, Credit History, and Property Area.

Data Preprocessing

Encoded categorical variables using Label Encoding.

Normalized and structured the dataset for model training.

Model Training

Chosen algorithm: Random Forest Classifier.

Performed GridSearchCV for hyperparameter tuning.

Saved trained model and encoders as .pkl files.

Deployment

Built an interactive Streamlit app.

Integrated trained model to accept user inputs and predict loan approval.

📊 Results

The Random Forest Classifier achieved an accuracy of around 78–80% on validation data.

The app predicts loan approval with high reliability based on applicant details.

The model successfully handles both categorical and numerical inputs.

📂 Project Structure
loan-approval-ml/
│── app.py                  # Streamlit app
│── loan_model.pkl          # Trained ML model
│── loan_encoders.pkl       # Encoders for categorical features
│── loan_target_encoder.pkl # Encoder for target variable
│── requirements.txt        # Dependencies
│── README.md               # Project description
│
└── data/
    └── loan_data.csv       # (Optional) Training dataset

🚀 Usage Instructions
1. Clone the repository
git clone https://github.com/your-username/loan-approval-ml.git
cd loan-approval-ml

2. Install dependencies
pip install -r requirements.txt

3. Run Streamlit app
streamlit run app.py

4. Open in browser

👉 Go to: http://localhost:8501

🌐 Deployment

Ready for deployment on Streamlit Cloud.

Upload files to GitHub → Connect repo on Streamlit Cloud
 → Deploy instantly.

📌 Conclusion

This mini-project demonstrates how Machine Learning can automate loan approval systems, reducing manual effort and decision bias.
It provides an easy-to-use web interface for banks and financial institutions to quickly assess applicants.

✨ Made with ❤️ using Python, scikit-learn, and Streamlit
