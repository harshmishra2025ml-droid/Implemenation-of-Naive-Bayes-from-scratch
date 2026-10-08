import pandas as pd

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv(r'C:\Users\Harsh Mishra\Documents\Data set\play_tennis.csv')

# Remove unwanted column
data.drop(columns=['day'], inplace=True)

# Convert all string values to lowercase and remove extra spaces
for col in data.columns:
    data[col] = data[col].astype(str).str.strip().str.lower()

# Display dataset
print("Dataset:")
print(data)

# -----------------------------
# Prior Probabilities
# -----------------------------
total = len(data)

yes_count = len(data[data['play'] == 'yes'])
no_count = len(data[data['play'] == 'no'])

py = yes_count / total
pn = no_count / total

print("\nPrior Probabilities")
print("P(Yes) =", py)
print("P(No)  =", pn)

# -----------------------------
# Conditional Probability Function
# -----------------------------
def cond_prob(feature, value, target):
    numerator = len(data[(data[feature] == value) & (data['play'] == target)])
    denominator = len(data[data['play'] == target])

    if denominator == 0:
        return 0

    return numerator / denominator

# -----------------------------
# Sample to Predict
# -----------------------------
sample = {
    'outlook': 'sunny',
    'temp': 'hot',
    'humidity': 'high',
    'wind': 'weak'
}

# -----------------------------
# Calculate P(Yes)
# -----------------------------
pyes = py

print("\nConditional Probabilities for YES")
for feature, value in sample.items():
    prob = cond_prob(feature, value, 'yes')
    print(f"P({feature}={value} | Yes) = {prob}")
    pyes *= prob

# -----------------------------
# Calculate P(No)
# -----------------------------
pno = pn

print("\nConditional Probabilities for NO")
for feature, value in sample.items():
    prob = cond_prob(feature, value, 'no')
    print(f"P({feature}={value} | No) = {prob}")
    pno *= prob

# -----------------------------
# Final Result
# -----------------------------
print("\nFinal Probabilities")
print("P(Yes) =", pyes)
print("P(No)  =", pno)

if pyes > pno:
    print("\nPrediction : YES")
else:
    print("\nPrediction : NO")