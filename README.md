# Selective Attention and Convolution: A Novel Approach to Sequence-Based Text Classification

## **Abstract**
Text classification plays a vital role in various applications such as e-commerce, search engines, and content categorization. This study proposes a novel hybrid framework that combines Convolutional Neural Networks (CNN) and Self-Attention mechanisms to address the challenges of sequence data classification. Through extensive ablation studies and comparisons with traditional models, the proposed architecture demonstrates improved efficiency and accuracy in handling text-based tasks.

---

## **1. Introduction**
### **1.1 Motivation**
- Text classification is fundamental for applications like search engines and e-commerce.
- Efficient classification allows for better organization and retrieval of articles and products.
  
### **1.2 Challenges**
- High computational cost in sequence processing.
- Traditional models like Naive Bayes, KNN, and Autoencoders fail to handle complex textual patterns effectively.

### **1.3 Contributions**
- Propose a hybrid framework combining CNN for local feature extraction and Self-Attention for global dependencies.
- Introduce a Selective Attention mechanism to improve feature focus.
- Conduct extensive ablation studies to validate the contributions of each component.

---

## **2. Related Work**
(C-PsyD: A Chinese Text Classification Model for Detecting Psychological Problems)[https://assets-eu.researchsquare.com/files/rs-5337854/v1/310abd99-1943-4c45-a55f-7e5d9b736b2b.pdf?c=1737116245]
-提出了一種名為 C-PsyD 的中文文本分類模型，用於心理問題的檢測。
-目的是結合CNN、BiGRU、Attention和Self-Attention，設計一個性能優越的文本分類模型。
-與其他六種模型進行性能比較，包括 FastText、TextCNN、Simple-RNN、LSTM、BiLSTM 和 ST_MFLC。
---

## **3. Methodology**
### **3.1 Model Architecture**
1. **Embedding Layer**: Converts input text into dense vector representations.
2. **Convolutional Layers**: Extract local patterns and n-gram features.
3. **Self-Attention Mechanism**: Captures global dependencies and highlights critical features.
4. **Selective Attention**: Filters noise and enhances important parts of the sequence.
5. **Fully Connected Layers**: Produces the final classification output.

### **3.2 Ablation Studies**
- **Combinations Tested**:
  1. CNN + Self-Attention.
  2. CNN only.
  3. Self-Attention only.
  4. No CNN, No Self-Attention (baseline).

---

## **4. Experiments**
### **4.1 Dataset**
- **E-commerce Product Reviews**: Categorization of product reviews into multiple classes.
- **Benchmark Text Datasets**: IMDB, Yelp Reviews, etc.

### **4.2 Evaluation Metrics**
- Accuracy, Precision, Recall, F1-score.

### **4.3 Comparative Analysis**
- **Baseline Models**: Naive Bayes, KNN, LSTM, and DNN.
- **Proposed Hybrid Framework**: Demonstrates superior performance in accuracy and efficiency.

---

## **5. Results**
- **Model Performance**:
  - CNN + Self-Attention achieved the highest F1-score (X%).
  - Selective Attention mechanism contributed to a Y% performance boost.
- **Ablation Study**:
  - Removing CNN led to a significant performance drop (-Z%).
  - Self-Attention independently showed limited results but combined with CNN improved outcomes.

---

## **6. Discussion**
- **Strengths**:
  - Efficient processing of sequence data.
  - Enhanced performance in capturing both local and global textual features.
- **Limitations**:
  - Higher computational cost than traditional methods.

---





