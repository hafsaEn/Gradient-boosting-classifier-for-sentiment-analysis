import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# Load the Sentiment140 training dataset
train_df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin-1', header=None)
train_df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']

# Load the Sentiment140 test dataset
test_df = pd.read_csv('testdata.manual.2009.06.14.csv', encoding='latin-1', header=None)
test_df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']

# Select the relevant columns
train_df = train_df[['text', 'target']]
test_df = test_df[['text', 'target']]

# Convert target to binary (0 for negative, 4 for positive)
train_df['target'] = train_df['target'].apply(lambda x: 1 if x == 4 else 0)
test_df['target'] = test_df['target'].apply(lambda x: 1 if x == 4 else 0)

# Combine train and test data for vectorization to ensure the same feature space
combined_df = pd.concat([train_df, test_df])
vectorizer = TfidfVectorizer(max_features=10000)  # Limit to top 10,000 features for efficiency
X_combined = vectorizer.fit_transform(combined_df['text'])

# Split the combined features back into train and test sets
X_train = X_combined[:len(train_df)]
X_test = X_combined[len(train_df):]

# Target variables
y_train = train_df['target']
y_test = test_df['target']

# Create and train the XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)
