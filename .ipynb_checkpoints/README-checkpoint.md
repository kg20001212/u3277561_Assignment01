# Predictive Modelling and Deployment – Assignment 1

**Student ID:** u3277561  

This project demonstrates an end-to-end data science workflow including:  
- Exploratory Data Analysis (EDA)  
- Predictive Modelling (Regression + Classification)  
- Reproducibility with Git & DVC  

---

##  Project Structure
```u3277561_Assignment01/
├── data/                 # Raw + processed datasets (DVC-tracked)
├── reports/              # Regression & classification results (DVC outputs)
├── src/                  # Python scripts
│   ├── preprocess.py
│   ├── features.py
│   ├── model.py
│   └── evaluate.py
├── dvc.yaml              # DVC pipeline definition
├── dvc.lock              # DVC pipeline lockfile
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation

```
## Installation
```bash
git clone <repo-url>
cd u3277561_Assignment01
py -m venv venv
venv\Scripts\activate
py -m pip install -r requirements.txt


