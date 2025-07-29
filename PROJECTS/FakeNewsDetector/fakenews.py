import pandas as pd

# Load datasets
df_fake = pd.read_csv('Fake.csv')
df_real = pd.read_csv('True.csv')

# Add labels
df_fake['label'] = 1  # fake
df_real['label'] = 0  # real

# Combine datasets
df = pd.concat([df_fake, df_real], axis=0)
df = df[['title', 'text', 'label']]
df = df.dropna()

# Combine title + text into one feature
df['content'] = df['title'] + " " + df['text']

# Show basic info
print("Dataset size:", df.shape)
print(df.head())



import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download stopwords only ONCE and store them globally
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))  # ✅ Move outside the function

ps = PorterStemmer()

# Function to clean and stem text
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stop_words]  # ✅ Use preloaded stop_words
    return ' '.join(text)


# Apply cleaning function
df['cleaned'] = df['content'].apply(clean_text)

# Show a sample of cleaned text
print(df[['content', 'cleaned']].head())

df.to_csv('cleaned_data.csv', index=False)



from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=5000)  # You can tune max_features

# Fit and transform the cleaned text
X = tfidf.fit_transform(df['cleaned']).toarray()

# Target variable
y = df['label']



from sklearn.model_selection import train_test_split

# Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))



import pickle

pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(model, open('model.pkl', 'wb'))
