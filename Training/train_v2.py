import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('../Data/train.csv')

def advanced_feature_engineering(df, is_train=True):
    df = df.copy()
    
    # Cabin features
    df['Deck'] = df['Cabin'].str.split('/').str[0]
    df['Side'] = df['Cabin'].str.split('/').str[2]
    df['CabinNumber'] = df['Cabin'].str.split('/').str[1].fillna('0').astype(int)
    
    # PassengerId features
    df['Group'] = df['PassengerId'].str.split('_').str[0]
    df['NumberInGroup'] = df['PassengerId'].str.split('_').str[1].astype(int)
    
    # Spending features
    spending_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalSpending'] = df[spending_features].sum(axis=1)
    df['HasSpending'] = (df['TotalSpending'] > 0).astype(int)
    df['SpendingRatio'] = df['RoomService'] / (df['TotalSpending'] + 1e-6)
    
    # Individual spending ratios
    for feature in spending_features:
        df[f'{feature}_Ratio'] = df[feature] / (df['TotalSpending'] + 1e-6)
    
    # Age features
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 30, 50, 100], 
                           labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior'])
    df['IsChild'] = (df['Age'] <= 12).astype(int)
    df['IsElderly'] = (df['Age'] >= 60).astype(int)
    
    # Group features (careful to avoid data leakage)
    if is_train:
        group_stats = df.groupby('Group').agg({
            'Age': ['mean', 'std', 'count'],
            'TotalSpending': 'mean'
        }).fillna(0)
        group_stats.columns = ['Group_Age_Mean', 'Group_Age_Std', 'Group_Size', 'Group_Spending_Mean']
        df = df.merge(group_stats, on='Group', how='left')
    else:
        # For test data, you'd need to handle this differently or use pre-computed stats
        pass
    
    # Interaction features
    df['VIP_Spending'] = df['VIP'] * df['TotalSpending']
    df['CryoSleep_Spending'] = df['CryoSleep'] * df['TotalSpending']
    
    # Drop original columns
    df.drop(['Cabin', 'Name'], axis=1, inplace=True)
    
    return df

# Apply feature engineering
train_df = advanced_feature_engineering(df)

# Prepare target
X = train_df.drop(columns=['Transported'])
y = train_df['Transported'].astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Identify feature types
numerical_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Preprocessing
numerical_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Try multiple models
models = {
    'xgb': XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False),
    'lgbm': LGBMClassifier(random_state=42, verbose=-1),
    'catboost': CatBoostClassifier(random_state=42, verbose=0),
    'gb': GradientBoostingClassifier(random_state=42)
}

# Train and evaluate each model
best_model = None
best_score = 0

for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Cross-validation
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    mean_score = scores.mean()
    print(f"{name}: {mean_score:.4f} (+/- {scores.std():.4f})")
    
    if mean_score > best_score:
        best_score = mean_score
        best_model = pipeline

# Fit best model
print(f"\nBest model: {best_score:.4f}")
best_model.fit(X_train, y_train)

# Evaluate
train_pred = best_model.predict(X_train)
test_pred = best_model.predict(X_test)

print(f"Train Accuracy: {accuracy_score(y_train, train_pred):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, test_pred):.4f}")

# Feature importance for the best model
if hasattr(best_model.named_steps['model'], 'feature_importances_'):
    feature_importances = best_model.named_steps['model'].feature_importances_
    # You can analyze which features are most important

# For test predictions
test_df = pd.read_csv('../Data/test.csv')
test_df_engineered = advanced_feature_engineering(test_df, is_train=False)

# Make sure test data has same columns as training
missing_cols = set(X.columns) - set(test_df_engineered.columns)
for col in missing_cols:
    test_df_engineered[col] = 0  # or appropriate default

test_df_engineered = test_df_engineered[X.columns]  # Ensure same column order

test_predictions = best_model.predict(test_df_engineered)
test_predictions_bool = test_predictions.astype(bool)

# Create submission
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Transported': test_predictions_bool
})

submission.to_csv('../Answer/answers_improved.csv', index=False)
print("Submission file created!")