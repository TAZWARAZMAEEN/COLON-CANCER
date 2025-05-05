# 🧠 Colon Cancer Cell Type Classification with CNN

This project uses a deep **Convolutional Neural Network (CNN)** to classify histopathological images of colon cancer cells into **four distinct cell types**:  
- **Epithelial**
- **Lymphocyte**
- **Goblet**
- **Stroma**

The dataset used is a modified version of **CRCHistoPhenotypes**, and the model is trained and evaluated using `TensorFlow` and `Keras`.

---

## 📂 Dataset

The dataset includes 4,000+ image samples of size **27x27 pixels**, each labeled with a specific cell type.

- `X_train`, `X_test`: Preprocessed images as NumPy arrays  
- `y_train`, `y_test`: Corresponding labels  

Data was augmented and normalized to enhance generalization and prevent overfitting.

---

## 🧪 Model Architecture

The CNN consists of:

- Conv2D + ReLU + MaxPooling layers  
- Dropout for regularization  
- Dense layers with softmax for final classification  

The model was trained using:
- `categorical_crossentropy` loss  
- `Adam` optimizer  
- Accuracy as the primary metric

---

## 📈 Results

- **Training Accuracy:** ~95%  
- **Validation Accuracy:** ~90%  
- **Confusion Matrix** and **Classification Report** show strong performance across all four classes.

---

## 🛠 Technologies Used

- Python  
- TensorFlow / Keras  
- NumPy, Matplotlib, scikit-learn  
- Google Colab (for training & experimentation)  

---

## 📊 Evaluation Metrics

- Confusion Matrix  
- Precision, Recall, F1-Score  
- Training vs. Validation Accuracy/Loss graphs  
- Class-wise prediction results  

---

## 📌 Key Highlights

- Applied **image augmentation** to improve generalization  
- Used **Dropout and L2 Regularization** to avoid overfitting  
- Performed **hyperparameter tuning** for CNN layers and batch size  
- Validated model performance using robust visualizations and metrics  

---

## 🚀 Future Improvements

- Use transfer learning with pre-trained models (e.g., ResNet, VGG)  
- Increase dataset size and resolution  
- Deploy the model using Flask or Streamlit for live inference  

---

## 🤝 Acknowledgements

- Dataset: [CRCHistoPhenotypes](https://zenodo.org/record/53169)  
- Tools: TensorFlow, Google Colab, scikit-learn  
- Mentors and study buddies from AI Study Group

---
