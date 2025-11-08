# Cyberbullying Tweet Multi-Class Classification

This repository contains a Natural Language Processing (NLP) project focused on identifying and classifying different types of **cyberbullying in tweets**. With the rapid rise of social media usage—especially during the COVID-19 pandemic—cyberbullying has become more pervasive and harmful. This project aims to support efforts in automatic detection systems that can flag hateful, abusive, and harmful tweets and help reduce the psychological damage caused by online harassment.

---

## 📌 Problem Statement

Social media has become a primary communication platform for people of all age groups. However, its widespread use has also intensified the prevalence and impact of cyberbullying, which can occur at any time and from anywhere. The relative anonymity of the internet enables individuals to engage in harmful behavior with fewer immediate consequences compared to traditional, face-to-face bullying, making it more difficult to detect and stop.

During the COVID-19 pandemic, the situation worsened significantly. On April 15th, 2020, UNICEF issued a warning regarding the heightened risk of online harassment due to increased screen time, reduced in-person social interaction, and widespread school closures. Statistics show that **36.5%** of middle and high school students have experienced cyberbullying, while **87%** have witnessed it. These incidents can lead to serious outcomes such as anxiety, depression, decreased academic performance, and even suicidal thoughts. This project aims to build a machine learning model that can automatically identify and categorize different types of cyberbullying to support early detection and digital safety efforts.

This project aims to:

- Develop a **multi-class classification model** that categorizes tweets based on the type of cyberbullying.
- Analyze **linguistic patterns** present in hateful or bullying content.
- Evaluate classification performance using **Weighted F1-score** due to class imbalance.

---

## 🧠 Features

The dataset contains the following features:

| Feature Name        | Description                                                   | Type        |
|---------------------|---------------------------------------------------------------|-------------|
| `tweet_text`        | The tweet content extracted from social media                 | Text        |
| `cyberbullying_type`| The annotated class label indicating type of cyberbullying    | Categorical |

---

## 📂 Project Structure

```bash
cyberbullying-classification-multiliingual-nlp/
├── data/
│   ├── raw/                        # Original dataset
│   │   ├── cyberbullying_tweets.csv
│   └── cleaned/                    # Cleaned and preprocessed dataset
│       ├── cyberbullying_tweets_cleaned.csv
├── notebooks/
│   └── eda.ipynb                   # Data exploration and preprocessing
│   └── model.ipynb                 # Model building and evaluation
├── models/
│   └── best_cyberbullying_model.pt # Saved trained model
├── requirements.txt                # Dependency file
├── README.md                       # Documentation
└── .gitignore
```

## 🔁 Workflow

This project follows a typical machine learning workflow:

1. Data Collection and Preparation
   - Downloaded from Kaggle (see [Dataset & Credits](#-dataset--credits) section).
   - Create train and test set from splitting the data.

2. Data Preprocessing
   - Corrected formatting inconsistencies and handled multilingual text.
   - Performed text cleaning: casefolding, demojizing, removing URLs, special characters, and stopwords.
   - Performed feature engineering and label encoding for `Target`.

3. Exploratory Data Analysis (EDA)
   - Analyzed `Target` distribution.
   - Analyzed word frequency and common phrases in each class.
   - Visualized correlations between features and the target.

4. Model Training
   - Tried multiple classification models: SVM, Random Forest, LightGBM, CatBoost, and XGBoost.
   <!-- - Implemented deep learning models: LSTM + GRU and Transformer-based (BERT). -->

5. Model Evaluation
   - Evaluated models using Weighted F1 Score, appropriate for imbalanced class distributions.
   - Created confusion matrix and detailed classification reports.
   - Best-performing model: ****.

## 🔤 Text Embedding

To convert the tweet text into numerical representations suitable for machine learning models (not included transformer model), this project uses **Sentence Transformer embeddings**.  
We specifically utilize the multilingual model:

```
paraphrase-multilingual-mpnet-base-v2
```
This model was selected because:
- It supports multiple languages, making it suitable for diverse Twitter content.
- It captures semantic meaning, not just word-level patterns.
- It is optimized for sentence-level similarity, allowing tweets with similar intent or tone to be represented closely in vector space.

## 📈 Model Performance

Several classification models were evaluated to categorize user statements into one of the seven mental health status labels.  
Model performance was measured using the **Weighted F1 Score**, which is suitable for imbalanced multi-class classification.  
The summarized results are shown below:

| Model                    | Weighted F1 Score |
|------------------------|------------------|
| Random Forest             | 76.95           |
| **CatBoost**              | **80.61**           |
| XGBoost                   | 79.99           |
| LightGBM                  | 80.36           |
| SVM                       | 53.52           |


The **CatBoost** achieved the **highest Weighted F1 Score**, indicating its ability to effectively handle imbalanced data and capture complex patterns in the text data.
Therefore, it was selected as the **final model** for inference.

---

## 📂 Dataset & Credits

The dataset used in this project was sourced from Kaggle.  
You can access the original dataset and description through the link below:

🔗[Cyberbullying Classification Dataset](https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification)

We would like to acknowledge and thank the dataset creator for making this resource publicly available for research and educational use.

---

## 🚀 How to Run

To run this project on your local machine, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/abidalfrz/cyberbullying-classification-multilingual-nlp.git
cd cyberbullying-classification-multilingual-nlp
```

### 2. Create and Activate a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate      # On Linux/macOS
venv\Scripts\activate.bat     # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Register the Virtual Environment as a Kernel (If using Jupyter Notebooks)

```bash
python -m ipykernel install --user --name name-kernel --display-name "display-name-kernel"
```

### 5. Run the Jupyter Notebook

Make sure you have Jupyter installed and select the kernel that you just created, then run the notebooks:

```bash
jupyter notebook notebooks/eda.ipynb
jupyter notebook notebooks/model.ipynb
```

You can explore:
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Model training and evaluation
- Generating final predictions

