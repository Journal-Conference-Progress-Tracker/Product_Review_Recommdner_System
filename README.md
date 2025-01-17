# Selective Attention and Convolution: A Novel Approach to Sequence-Based Text Classification

## **Abstract**
Text classification.....

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
[C-PsyD: A Chinese Text Classification Model for Detecting Psychological Problems](https://assets-eu.researchsquare.com/files/rs-5337854/v1/310abd99-1943-4c45-a55f-7e5d9b736b2b.pdf?c=1737116245)
- 提出了一種名為 C-PsyD 的中文文本分類模型，用於心理問題的檢測(多元分類)。
- 目的是結合CNN、BiGRU、Attention和Self-Attention，設計一個性能優越的文本分類模型。
- 與其他六種模型進行性能比較，包括 FastText、TextCNN、Simple-RNN、LSTM、BiLSTM 和 ST_MFLC。

[Cyberbullying Detection in Social Networks Using Deep Learning](https://www.iajit.org/upload/files/Cyberbullying-Detection-in-Social-Networks-Using-Deep-Learning.pdf)
- 使用Facebook data，包含 11,000 條已標記的評論（分為乾淨或霸凌）(二元分類)
- 比較CNN、LSTM、CNN-LSTM、BERT
- BERT 最佳，CNN-LSTM第二

[malDetect: Malware Classification Using API Sequence and Comparison with Transformer Encoder](https://ieeexplore.ieee.org/abstract/document/10731782)
- 提出了一種基於CNN-LSTM的改進模型來檢測和分類惡意軟體程序
- 使用的資料是動態運行後擷取的Windows API序列。文字處理方法也適用於處理序列資料
- 該模型將 BiLSTM 模型與 Self-Attention 機制結合，命名為 malDetect II，比基本模型 CNN-LSTM 提升了 11.46%，比 Transformer Encoder 分類模型提升了 2.82%。
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





