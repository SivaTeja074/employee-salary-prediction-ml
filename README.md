<p align="center">
  <img src="assets/cover.png" alt="Employee Salary Prediction Banner" width="800"/>
</p>

# 💼 Employee Salary Prediction Using Machine Learning

![GitHub last commit](https://img.shields.io/github/last-commit/SivaTeja074/employee-salary-prediction-ml)
![GitHub repo size](https://img.shields.io/github/repo-size/SivaTeja074/employee-salary-prediction-ml)
![GitHub stars](https://img.shields.io/github/stars/SivaTeja074/employee-salary-prediction-ml?style=social)
![Python](https://img.shields.io/badge/Made%20with-Python-blue)

Predict whether an employee earns more than \$50K per year using demographic and work-related attributes, with a fully deployed interactive web app.


## 📂 Project Structure

employee-salary-prediction-ml/
│
├── data/
│   └── employee_salary_dataset.csv         # Raw dataset
│
├── notebooks/
│   └── Employee_salary_prediction.ipynb    # Jupyter Notebook for EDA & training
│
├── src/
│   ├── train_model.py                      # Python script to train and export model
│   ├── app.py                              # Streamlit app for deployment
│
├── assets/
│   ├── cover.png                           # Professional project banner for top of README
|   |── Model_Comparison.png                # Model accuracy bar graph
│   ├── Screenshot_1.png                    # App input form UI
│   ├── Screenshot_2.png                    # App input form bottom
│   ├── Screenshot_3.png                    # CSV upload for bulk prediction
│   ├── Screenshot_4.png                    # Download CSV result
│   ├── Screenshot_5.png                    # Bulk output CSV open
│   ├── Screenshot_6.png                    # Final predictions in Excel
│
├── best_model.pkl                          # Saved trained ML pipeline
├── requirements.txt                        # Python dependencies
├── .gitignore                              # Ignore rules
|── LICENSE                                 # MIT License 
├── README.md                               # Project documentation (this file)


✅ Project Objective

📌 Goal: Classify whether an employee’s annual income exceeds $50K based on socio-economic data.

📌 Dataset: Census-like data with multiple demographic & work attributes.

📌 Pipeline: Data cleaning ➜ Feature engineering ➜ Multiple ML models ➜ Best model ➜ Deployed with Streamlit.


⚙️ Tech Stack

Language: Python 3.x

Libraries: pandas, scikit-learn, joblib, streamlit, pyngrok

Deployment: Local Streamlit server + optional Ngrok tunnel for public access


🔎 Model Performance

Model	Accuracy
Logistic Regression	0.84
Random Forest	0.84
K-Nearest Neighbors	0.82
Support Vector Machine (SVM)	0.85
Gradient Boosting	0.86 ✅ (Best)

<p align="center"> <img src="assets/Model_Comparison.png" alt="Model Accuracy Comparison" width="600"/> </p>
➜ Five ML algorithms evaluated

➜ GradientBoosting selected as final best model

➜ Preprocessing: OneHotEncoder for categoricals, StandardScaler for numericals


🚀 How to Run Locally

1️⃣ Clone the repo

bash
Copy
Edit
git clone https://github.com/SivaTeja074/employee-salary-prediction-ml.git
cd employee-salary-prediction-ml

2️⃣ Install dependencies

bash
Copy
Edit
pip install -r requirements.txt

3️⃣ Train the model (creates best_model.pkl)

bash
Copy
Edit
python src/train_model.py

4️⃣ Run the Streamlit app

bash
Copy
Edit
streamlit run src/app.py
Open http://localhost:8501 in your browser ✅


✅ Screenshots

🎯 Input Form
<p align="center"> <img src="assets/Screenshot_1.png" alt="App Input Form" width="700"/> </p>

🎯 Input Bottom & Prediction
<p align="center"> <img src="assets/Screenshot_2.png" alt="App Prediction" width="700"/> </p>

🎯 Bulk CSV Upload
<p align="center"> <img src="assets/Screenshot_3.png" alt="Bulk Upload" width="700"/> </p>

🎯 Download CSV Result
<p align="center"> <img src="assets/Screenshot_4.png" alt="Download Output" width="700"/> </p>

🎯 Output CSV Open
<p align="center"> <img src="assets/Screenshot_5.png" alt="Output CSV" width="700"/> </p>

🎯 Final Salary Predictions
<p align="center"> <img src="assets/Screenshot_6.png" alt="Final CSV Predictions" width="700"/> </p>


📈 Results

✅ Trained model (best_model.pkl) reused for live predictions.

✅ Real-time input + batch CSV supported.

✅ Final predictions downloadable as CSV.



🔒 Deployment Note

Use pyngrok or your Ngrok CLI to tunnel localhost:8501 for a public link.

Example:

bash
Copy
Edit
ngrok http 8501


🔮 Future Scope

Deploy to Streamlit Cloud, Render, or Heroku for permanent public hosting.

Use a database to log predictions.

Expand dataset with additional factors.

Add authentication for secure user access.



📚 References

scikit-learn Documentation

Streamlit Docs

UCI Machine Learning Repository: Adult Dataset

Your dataset: data/employee_salary_dataset.csv


## 📜 License

This project is licensed under the **MIT License** 

See the full LICENSE file for details.


🤝 Author

Name: Siva Teja Talari

GitHub: github.com/SivaTeja074


## 🤝 Contributions

Contributions, issues, and feature requests are welcome!  

