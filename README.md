# Student Performance Prediction System

A classroom prototype built with Python, Random Forest, and Streamlit. It uses the UCI Student Performance dataset to estimate whether a record belongs to the at-risk group.

## Prediction rule

- **AT RISK:** G3 is below 10.
- **NOT AT RISK:** G3 is 10 or higher.
- **Important:** G3 is used only to create the training target. It is never used as an input feature.

## Dataset

UCI Machine Learning Repository — Student Performance:
https://archive.ics.uci.edu/dataset/320/student+performance

The project downloads the dataset using the `ucimlrepo` Python package.

## Technologies

- Python
- pandas
- scikit-learn
- Random Forest
- Streamlit
- GitHub

## Run on your computer

1. Install Python 3.10 or later.
2. Download or clone this repository.
3. Open a terminal in the project folder.
4. Install the packages:

   ```bash
   pip install -r requirements.txt
   ```

5. Train the model and create the dataset/model files:

   ```bash
   python train_model.py
   ```

6. Start the Streamlit application:

   ```bash
   streamlit run app.py
   ```

7. Open the local URL shown in the terminal, usually `http://localhost:8501`.

The app can also download the dataset and train the model automatically on its first run if the generated files are not present.

## Upload to GitHub

Create a repository named `student-performance-prediction`. Upload these files:

- `app.py`
- `model_utils.py`
- `train_model.py`
- `requirements.txt`
- `README.md`
- `.gitignore`

You do not need to upload `student_data.csv` or `student_model.pkl`; the app can create them when it starts. Do not upload passwords, access tokens, or private student information.

## Deploy with Streamlit Community Cloud

1. Push this project to GitHub.
2. Sign in to Streamlit Community Cloud using your GitHub account.
3. Select **Create app**.
4. Choose this repository, the `main` branch, and `app.py` as the main file.
5. Select **Deploy** and wait for the app to build.
6. Open the app URL provided by Streamlit.

The first startup needs internet access to retrieve the public UCI dataset and train the model.

## System users and SAD summary

- **Problem:** Academic risk may only be noticed after poor results.
- **Users:** Faculty members or academic analysts.
- **Inputs:** Student attributes and academic indicators such as study time, absences, past failures, G1, and G2.
- **Process:** Validate data, preprocess it, apply the trained model, and classify the record.
- **Outputs:** AT RISK / NOT AT RISK prediction and dataset/model information.
- **Limitation:** A model trained on one Portuguese secondary-school dataset may not work equally well for another school or population.

## Responsible use

This is a classroom demonstration, not a diagnosis or a final decision-making tool. A faculty member should review the student's circumstances and use appropriate judgment. Do not use predictions as the sole basis for real student decisions.
