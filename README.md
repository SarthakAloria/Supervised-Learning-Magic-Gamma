[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)  [![Scikit-Learn](https://img.shields.io/badge/scikit--learn-0.24%2B-orange)](https://scikit-learn.org/stable/)  [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-ff6f00)](https://www.tensorflow.org/)  [![UCI Dataset](https://img.shields.io/badge/Dataset-UCI%20MAGIC%20Gamma-brightgreen)](https://archive.ics.uci.edu/ml/datasets/magic+gamma+telescope)  
# Comparing Supervised Learning Algorithms on the MAGIC Gamma Telescope Dataset

##  Project Description

This project performs a **comparative analysis of supervised learning algorithms** on the **MAGIC Gamma Telescope dataset** from UCI Machine Learning Repository. The dataset contains features extracted from gamma-ray and hadron events detected by the telescope. The goal is to classify events as gamma or hadron using multiple machine learning models and evaluate their performance.

The project implements and compares the following supervised learning algorithms:  

- **K-Nearest Neighbors (KNN)**  
- **Naive Bayes**  
- **Logistic Regression**  
- **Support Vector Machine (SVM)**  
- **Neural Networks**  

Evaluation is done based on metrics such as **accuracy, precision, recall, F1-score**, and **loss curves**. The project provides insights into which algorithm performs best on this dataset and highlights trade-offs between model complexity and performance.


**Key Details:**
- **Dataset:** [UCI MAGIC Gamma Telescope Dataset](https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope)  
- **Problem Type:** Binary Classification  
- **Programming Language:** Python  
- **Libraries Used:** scikit-learn, pandas, numpy, matplotlib, seaborn, tensorflow/keras  
- **Approach:** Train multiple supervised models → Evaluate metrics → Compare performance  
- **Outcome:** Ranked performance of algorithms and visual analysis of predictions

---
## Table of Contents

- [Features](#features)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Conclusion](#conclusion)
- [Tech Stack](#tech-stack)
- [Contributing](#contributing)
- [Authors](#authors)
- [License](#license)

---
##  Features

- Implements multiple **supervised learning algorithms** for binary classification: KNN, Naive Bayes, Logistic Regression, SVM, and Neural Networks.  
- Performs **comparative analysis** of model performance using key metrics like accuracy, precision, recall, and F1-score.  
- Visualizes results with **confusion matrices, loss curves, and performance charts** for easy interpretation.  
- Provides insights on **algorithm strengths, weaknesses, and trade-offs** for this specific dataset.  
- Structured for **reproducibility**, making it easy to extend or apply to similar classification problems.  


---
## Dataset

- **Name:** MAGIC Gamma Telescope Dataset  
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope)  
- **Description:** The dataset contains **high-level features extracted from images** captured by the MAGIC Cherenkov telescope. Each instance represents an event classified as **gamma-ray** or **hadron**.  
- **Number of Instances:** 19,020  
- **Number of Features:** 10 numeric features per instance  
- **Target Variable:** `class` – indicates whether the event is a gamma-ray (`g`) or hadron (`h`)  
- **Usage:** Used for **binary classification** tasks to distinguish gamma-ray events from background hadrons.  

---
## Methodology
The project follows a structured approach to classify events in the MAGIC Gamma Telescope dataset:

1. **Data Preprocessing**
   - Loaded the dataset and handled any missing or inconsistent values.
   - Standardized numeric features to ensure uniform scaling for algorithms sensitive to feature magnitude.
   - Split the data into **training and testing sets** to evaluate model performance objectively.

2. **Model Implementation**
   - Implemented multiple **supervised learning algorithms**:
     - **K-Nearest Neighbors (KNN)**
     - **Naive Bayes**
     - **Logistic Regression**
     - **Support Vector Machine (SVM)**
     - **Neural Networks**
   - Tuned **hyperparameters** where applicable to optimize model performance.

3. **Model Evaluation**
   - Used key metrics to evaluate and compare models:
     - **Accuracy**
     - **Precision**
     - **Recall**
     - **F1-score**
   - Generated **confusion matrices** and **loss curves** for visual performance analysis.

4. **Comparative Analysis**
   - Ranked models based on evaluation metrics.
   - Discussed trade-offs between **model complexity** and **classification performance**.
   - Provided recommendations on the best-performing algorithm for this dataset.

---
## Evaluation Metrics

To assess and compare the performance of the supervised learning algorithms, the following metrics were used:

- **Accuracy:** Measures the proportion of correctly classified events out of the total events.  
- **Precision:** Indicates the proportion of correctly predicted positive events (gamma-rays) among all predicted positive events.  
- **Recall (Sensitivity):** Measures the proportion of correctly predicted positive events out of all actual positive events.  
- **F1-Score:** The harmonic mean of precision and recall, providing a balance between the two.  
- **Confusion Matrix:** Provides a detailed breakdown of true positives, true negatives, false positives, and false negatives for each model.  
- **Loss Curves:** For Neural Networks, visualizing training and validation loss over epochs to monitor convergence and potential overfitting.

These metrics provide a **comprehensive view of each model’s classification ability** and help identify the best-performing algorithm for the dataset.

---

## Results

### Model Training Curves
The following plots show the **training and validation curves** for the different models. Each image contains both **loss** and **accuracy** curves, demonstrating the effect of different hyperparameters such as **number of nodes** and **epochs**.

![Model 1 Curves](/model_training_curves/Model1.png)
![Model 2 Curves](/model_training_curves/Model2.png)

<p><strong>Note:</strong> Curves for all models are visually similar, indicating stable training across architectures.</p>

### Classification Performance
The overall performance of the models is summarized in the **classification report**, including accuracy, precision, recall, and F1-score.

![Result](/result/evaluation.png)


### Data Distribution
The following histogram shows the **distribution of gamma-ray and hadron events** in the dataset, which helps explain model behavior and class imbalance considerations.

![Class Distribution](/class_distributions/variable_table.png)

### Summary
- Training curves show that all models **converged effectively**, with minimal overfitting.  
- The classification report highlights the **best-performing models** and trade-offs between precision and recall.  
- The data histogram provides context for **model evaluation**, illustrating any class imbalance that might affect predictions.


---
## Conclusion

This project demonstrates a **comparative analysis of supervised learning algorithms** on the MAGIC Gamma Telescope dataset. Key takeaways include:

- **Neural Networks** provided the highest overall accuracy, but simpler models like **Logistic Regression** and **SVM** also performed competitively, demonstrating that model complexity should be balanced with dataset size and interpretability.  
- The evaluation metrics highlight the trade-offs between **precision** and **recall**, showing that different algorithms may favor detecting gamma-ray events versus hadron events.  
- Visualizations of **training/validation curves** help identify potential overfitting and ensure that models generalize well on unseen data.  
- The project is **reproducible and extensible**, making it easy to test additional algorithms, tune hyperparameters further, or apply the methodology to similar classification tasks in physics or astronomy datasets.

Overall, this work provides insights into **which supervised learning algorithms are best suited** for distinguishing gamma-ray events from background hadrons in this dataset, and serves as a reference for future machine learning applications in scientific data analysis.

---
## Tech Stack

The project was implemented using **Python** and several popular libraries for machine learning, data processing, and visualization:

- **Programming Language:** Python 3.x  
- **Data Handling:** pandas, numpy  
- **Machine Learning:** scikit-learn, tensorflow/keras  
- **Data Visualization:** matplotlib, seaborn  
- **Environment:** Jupyter Notebook / Google Colab  
- **Version Control:** Git & GitHub  

> This tech stack ensures **reproducibility, scalability, and ease of experimentation** with different algorithms and hyperparameters.
---
##  Contributing

Contributions are welcome! Whether it’s **improving model performance, adding new algorithms, optimizing data preprocessing, or enhancing visualizations**, you can help improve this project.

### How to Contribute

1. **Fork the repository**  
2. **Clone your fork** locally:
```bash
git clone https://github.com/SarthakAloria/supervised-learning-magic-gamma.git
```
3. **Create a new branch for your feature or bugfix:**
```bash
git checkout -b feature/your-feature-name
```
4. **Make your changes (e.g., add new models, improve preprocessing, add visualizations)**
5. **Commit your changes:**
```bash
git commit -m "Add: brief description of changes"
```
6. **Push your branch:**
```bash
git push origin feature/your-feature-name
```
7. **Open a Pull Request on the main repository.**

---
## Authors

- **Sarthak Aloria** – [SarthakAloria](https://github.com/SarthakAloria)


---
## Acknowledgements

- Dataset provided by the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope).  
- Inspiration and guidance from online tutorials, documentation, and machine learning communities.  
- Tools and libraries that made this project possible: **scikit-learn, TensorFlow, pandas, NumPy, matplotlib, and seaborn**.  
- Special thanks to mentors and peers who provided feedback during the project development.

---
## Support

For any issues or feature requests, please open an issue on GitHub or contact me at sarthakaloria27@gmail.com.

---