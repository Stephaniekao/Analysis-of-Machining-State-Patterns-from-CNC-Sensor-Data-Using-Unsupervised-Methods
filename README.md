# Analysis of Machining State Patterns from CNC Sensor Data Using Unsupervised Methods
# CNC Machining State Analysis Using Unsupervised Learning

Unsupervised learning framework for analyzing CNC machining behavior using multivariate sensor time-series data.

This project explores how clustering methods can identify stable, boundary, and unstable machining states without relying on labeled training data.

The workflow combines:

- time-series preprocessing
- machining-stage filtering
- sliding window segmentation
- statistical feature extraction
- first-level clustering
- second-level experiment clustering
- NMI-based validation and model comparison

The project was developed for EECE 5644 Machine Learning and Pattern Recognition at Northeastern University.

---

# 📦 Dataset Source

This project uses the publicly available CNC milling dataset from the University of Michigan SMART Lab.

Dataset:
CNC Milling Tool Wear Detection Dataset

Kaggle:
https://www.kaggle.com/datasets/shasun/tool-wear-detection-in-cnc-mill

The dataset contains:

- multivariate CNC sensor signals
- machining process labels
- tool condition information
- machining completion results
- visual inspection results

The original dataset includes 18 machining experiments under different machining conditions.

---

# 🎯 Project Objectives

This project aims to:

- analyze CNC machining behavior using unsupervised learning
- reduce dependence on labeled manufacturing data
- identify hidden machining state patterns
- distinguish stable, unstable, and boundary machining behaviors
- compare multiple clustering approaches
- explore intelligent manufacturing and process monitoring applications

---

# 🧠 Methodology

## 1️⃣ Data Preprocessing

The raw CNC time-series data includes both machining and non-machining periods.

To improve clustering quality, the preprocessing pipeline:

- removes non-machining periods
- filters abnormal rows
- separates machining stages
- constructs derived error features

### Machining stages retained

- Layer 1 Up
- Layer 1 Down
- Layer 2 Up
- Layer 2 Down
- Layer 3 Up
- Layer 3 Down

### Abnormal rows removed

Rows matching abnormal conditions defined in the original dataset documentation were removed to improve reliability.

---

## 2️⃣ Window-Based Representation

The time-series data is segmented into fixed-length windows.

Window sizes tested:

- 10
- 20
- 30

Additional settings:

- 50% overlap between windows
- process-specific segmentation
- short valid segments preserved if length ≥ 50% of window size

This allows the model to preserve temporal information while generating fixed-size representations.

---

## 3️⃣ Feature Extraction

Each window is transformed into a statistical feature vector.

### Statistical features

- mean
- standard deviation
- maximum
- minimum
- range
- root mean square (RMS)
- mean absolute difference

### Derived features

Additional control-error features were constructed:

- position error
- velocity error
- acceleration error

These features are calculated using:

Actual Value − Command Value

The derived features help represent machining stability and system deviation more effectively.

---

## 4️⃣ Clustering Models

Three unsupervised learning approaches were compared.

### K-Means

- cluster range: 2–9
- selected using silhouette score

### Gaussian Mixture Model (GMM)

- cluster range: 2–9
- selected using Bayesian Information Criterion (BIC)

### Self-Organizing Map (SOM)

Tested grid structures:

- 2 × 1
- 3 × 1
- 4 × 1
- 2 × 2

---

## 5️⃣ Second-Level Clustering

After first-level clustering:

- cluster distributions were computed for each experiment
- experiment-level vectors were constructed
- second-level clustering was performed using K-Means

Final experiment categories:

- stable behavior
- boundary behavior
- unstable behavior

---

## 6️⃣ Validation & Model Comparison

The clustering results were evaluated using:

- machining completion
- inspection outcome
- tool condition

### Evaluation Metric

Normalized Mutual Information (NMI)

Weighted final score:

- machining finalized → 0.6
- inspection result → 0.2
- tool condition → 0.2

---

# 📊 Results

## Best Model

Final selected model:

| Model | Window Size | Final Score |
|---|---|---|
| SOM | 30 | 0.494 |

---

## Key Findings

### Stable behavior
- completed machining
- passed inspection

### Unstable behavior
- unfinished machining
- process instability

### Boundary behavior
- completed machining
- failed inspection
- unstable product quality

The results demonstrate that unsupervised learning can distinguish different levels of machining stability without labeled training data.

---

# 🧩 Project Features

✔ Dynamic time-series preprocessing

✔ Process-specific window segmentation

✔ Statistical feature engineering

✔ Derived control-error features

✔ Multiple clustering model comparison

✔ Two-stage clustering framework

✔ NMI-based validation

✔ Experiment-level behavior analysis

✔ Smart manufacturing application potential

---

# 🚀 Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- seaborn
- MiniSom

---

# 📄 Full Report

See the complete academic report here:

[Analysis of Machining State Patterns from CNC Sensor Data Using Unsupervised Methods.pdf)

---

# 🔥 Why This Project Is Interesting

Unlike traditional supervised manufacturing systems, this project:

- does not require labeled training data
- analyzes real CNC sensor signals
- captures hidden machining behavior patterns
- combines time-series analysis with clustering
- introduces second-level clustering for experiment-level interpretation

This framework has potential applications in:

- smart manufacturing
- process monitoring
- predictive maintenance
- quality control
- industrial AI systems

---

# 🚀 Future Improvements

Possible future extensions include:

- deep learning approaches
- real-time monitoring systems
- anomaly detection models
- digital twin integration
- larger industrial datasets
- online clustering systems

---

# 👩‍💻 Author

Wan Chi Kao

Master of Science in Artificial Intelligence  
Northeastern University

Research interests:

- machine learning
- smart manufacturing
- reliability engineering
- industrial AI
- intelligent systems

---

# 📜 License

This project uses publicly available datasets from Kaggle.

All rights belong to the original dataset creators.
