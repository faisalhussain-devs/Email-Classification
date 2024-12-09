# Spam Email Classifier Project

This project aims to build an efficient spam email classifier using machine learning techniques. Below is a detailed breakdown of the preprocessing steps, model evaluation, and results achieved in the project.

## **Data Preprocessing**

To ensure high-quality features for classification, the following preprocessing steps were applied to the email dataset:

1. **Text Normalization**:
   - Emails were converted to **lowercase** to ensure case consistency.
   - **Punctuation** was removed to focus on the text content.
   - **URLs** were replaced with the placeholder `"URL"` to generalize web links.
   - **Numbers** were replaced with the placeholder `"NUMBER"` to generalize numerical data.

2. **Lemmatization**:
   - Words were lemmatized using a lemmatizer, which performed better than stemming in this context by preserving grammatical correctness and semantic meaning.

3. **Feature Representation**:
   - **Vocabulary Creation**: A vocabulary was created based on the most frequent words across the training set.
   - **Sparse Matrix Encoding**: Emails were transformed into a **Compressed Sparse Row (CSR) matrix**, where each entry represents the presence (`1`) or absence (`0`) of a word from the vocabulary in the email.

4. **TF-IDF Transformation**:
   - For certain models (e.g., SVC, Random Forest, KNN), applying **TF-IDF (Term Frequency-Inverse Document Frequency)** transformation significantly improved results by emphasizing important words while de-emphasizing commonly occurring ones.

## **Models Tested**

Several machine learning models were evaluated for their performance on the spam classification task:

1. **K-Nearest Neighbors (KNN)**
2. **Naïve Bayes (MultinomialNB)**
3. **Logistic Regression**
4. **Random Forest Classifier**
5. **Support Vector Classifier (SVC)**

## **Model Optimization**

### **RandomizedSearchCV for Hyperparameter Tuning**
- Hyperparameter tuning was performed using **RandomizedSearchCV** to identify the best parameters for models such as Logistic Regression, Random Forest, and SVC.
- The optimized parameters significantly improved the performance metrics for these classifiers.

## **Results**

- **Best Models**:
  - SVC, Logistic Regression, and Random Forest achieved the highest scores, demonstrating robust performance on the processed dataset.

- **Selected Model**: After tuning and evaluation, **SVC** was selected as the final model based on its ability to maximize spam recall while maintaining good precision.
  - **Metrics for SVC**:
    - **Recall**: 99.16% (important to classify as many spam emails as possible).
    - **Precision**: 97.78% (minimizing false positives).

## **Challenges and Observations**

1. **TF-IDF vs. Word Count Representation**:
   - For models like SVC, Random Forest, and KNN, using **TF-IDF representation** yielded significantly better results compared to simple word count vectors.
   - However, for Logistic Regression and Naïve Bayes, the word count vector representation performed better.

2. **Class Imbalance**:
   - The dataset contained imbalances between spam and ham emails. Techniques like class weighting and careful metric selection (e.g., `recall` over `accuracy`) were used to address this.

3. **Hard Ham Emails**:
   - Emails categorized as "hard ham" (often containing more HTML content) posed challenges. Additional preprocessing techniques, such as stripping HTML tags and analyzing email structure, are being considered for further improvements.

## **Future Work**

1. **Handling Hard Ham**:
   - Implement advanced preprocessing techniques for HTML content, such as parsing and feature extraction (e.g., text-to-HTML ratio, number of links).

2. **Ensemble Models**:
   - Combine multiple models, such as Random Forest and SVC, to leverage their strengths for better overall performance.

3. **Feature Engineering**:
   - Incorporate custom features like spam keyword frequency, email length, and header analysis to improve classification accuracy.

4. **Deployment**:
   - Package the trained model into a deployable application for real-time spam detection.

This project highlights the effectiveness of preprocessing, feature engineering, and careful model selection in achieving state-of-the-art spam classification performance.

# Email-Classification
