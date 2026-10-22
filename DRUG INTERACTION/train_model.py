import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv('db_drug_interactions.csv', nrows=25000)
df = df.dropna()

if len(df) > 10000:
    df = df.sample(n=10000, random_state=42)

le_drug1 = LabelEncoder()
le_drug2 = LabelEncoder()
le_target = LabelEncoder()

df['drug_1_enc'] = le_drug1.fit_transform(df['Drug 1'].astype(str))
df['drug_2_enc'] = le_drug2.fit_transform(df['Drug 2'].astype(str))
df['target_enc'] = le_target.fit_transform(df['Interaction Description'].astype(str))

X = df[['drug_1_enc', 'drug_2_enc']].astype(np.int32)
y = df['target_enc'].astype(np.int32)

del df
gc.collect()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=20, max_depth=15, n_jobs=-1, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

joblib.dump(model, 'ddi_model.pkl')
joblib.dump(le_drug1, 'le_drug1.pkl')
joblib.dump(le_drug2, 'le_drug2.pkl')
joblib.dump(le_target, 'le_target.pkl')

print("Training finished successfully without crash!")