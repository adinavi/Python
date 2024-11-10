# train_model.py
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from data_preprocessing import load_data, preprocess_data

# Load and preprocess data
data = load_data('data/student_performance.csv')
data = preprocess_data(data)

# Split data
X = data[['study_hours', 'attendance']]  # Example features
y = data['grade']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save or print results
print("Model training complete. Score:", model.score(X_test, y_test))
