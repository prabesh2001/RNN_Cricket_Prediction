### **Description for GitHub Repository**

---

# **Cricket Series Outcome Prediction Using RNN**

This project implements a **Recurrent Neural Network (RNN)** to predict the outcomes of cricket series based on historical data. The dataset consists of features like participating teams, match counts, and series outcomes. The model utilizes TensorFlow/Keras to handle sequential data and provide accurate predictions.

---

## **Features**
- **Sequential Data Handling**: Transforms match data into time-series sequences for RNN input.
- **Classification Model**: Predicts outcomes (e.g., "Won," "Lost," "Drawn") of cricket series.
- **Performance Metrics**:
  - Precision, Recall, F1 Score.
  - AUC-ROC Curve (for binary classification).
- **Visualization**: Includes training accuracy/loss curves and ROC curve (if applicable).

---

## **Dataset**
The dataset contains the following features:
- **Team**: Name of the team.
- **Opponent**: Opposing team.
- **Matches**: Number of matches played.
- **Won, Lost, Drawn**: Statistics for the series.
- **Result**: Outcome of the series (Target Variable).

The dataset is preprocessed to encode categorical variables and normalize numerical features. Time-series sequences are generated to represent historical data.

---

## **How It Works**
1. **Data Preprocessing**:
   - Categorical variables (`Team`, `Opponent`, `Result`) are label-encoded.
   - Numerical variables are scaled using Min-Max scaling.
   - Time-series sequences of length 5 are created for input.
2. **Model Architecture**:
   - RNN with 64 units and ReLU activation.
   - Dropout layers to prevent overfitting.
   - Dense layers for classification with softmax activation.
3. **Training**:
   - Optimized using Adam.
   - Sparse categorical cross-entropy loss function.
4. **Evaluation**:
   - Accuracy, Precision, Recall, F1 Score, and AUC-ROC are computed.
5. **Predictions**:
   - Predicts the outcome of a cricket series based on the last 5 sequences.

---

## **Setup Instructions**
### Prerequisites
- Python 3.x
- Libraries: `tensorflow`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/cricket-series-rnn.git
   cd cricket-series-rnn
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Add your dataset:
   - Place the dataset (`testSeries.csv`) in the root directory.

---

## **Usage**
1. Run the preprocessing and model training script:
   ```bash
   python cricket_series_rnn.py
   ```
2. Evaluate the model and view performance metrics:
   - Classification Report: Precision, Recall, F1 Score.
   - AUC-ROC Curve (for binary classification).
3. Save or load the trained model for future use.

---

## **Repository Structure**
```plaintext
.
├── cricket_series_rnn.py      # Main script for preprocessing, model training, and evaluation
├── testSeries.csv             # Dataset file
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
```

---

## **Snapshots**
### Example Outputs:
1. **Dataset Preview**:
   ![Dataset Preview](images/dataset_preview.png)
2. **Training Accuracy/Loss**:
   ![Training Graph](images/training_graph.png)
3. **Classification Report**:
   ![Classification Report](images/classification_report.png)
4. **ROC Curve**:
   ![ROC Curve](images/roc_curve.png)

---

## **License**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## **Contributing**
Pull requests are welcome! For major changes, please open an issue to discuss your proposed contributions.

---

