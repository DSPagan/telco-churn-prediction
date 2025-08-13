from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

import pandas as pd

# Cargar y limpiar dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.drop(columns=["customerID"], inplace=True)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

# Limpiar y convertir TotalCharges
value = []
for i, elem in enumerate(df["TotalCharges"]):
    try:
        float(elem)
    except:
        value.append((i, elem))

df = df.drop([i for i, _ in value])
df["TotalCharges"] = df["TotalCharges"].astype("float64")
df.reset_index(drop=True, inplace=True)

# Separar X e y (target binarizado)
X = df.drop(columns=['Churn'])  # con mayúscula según dataset original
y = df['Churn'].map({'No':0, 'Yes':1})

# Columnas categóricas a binarizar (sin target)
cols_num = ['tenure', 'MonthlyCharges', 'TotalCharges']
cols_bin = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
cols_cat = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaymentMethod']

# Separar train (70%) y temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Separar eval (15%) y test (15%) del temp
X_eval, X_test, y_eval, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# ColumnTransformer configurado sin churn
ct = ColumnTransformer(
    transformers=[
        ('bin', OneHotEncoder(drop='if_binary', sparse_output=False, handle_unknown='ignore'), cols_bin),
        ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cols_cat),
        ('num', StandardScaler(), cols_num)
    ],
    remainder='drop'
)

# Fit solo en train
ct.fit(X_train)

# Transformar datasets
X_train_processed = ct.transform(X_train)
X_eval_processed = ct.transform(X_eval)
X_test_processed = ct.transform(X_test)

# nombres columnas
onehot_bin_cols = ct.named_transformers_['bin'].get_feature_names_out(cols_bin)
onehot_cat_cols = ct.named_transformers_['onehot'].get_feature_names_out(cols_cat)
num_cols_scaled = cols_num

final_columns = list(onehot_bin_cols) + list(onehot_cat_cols) + num_cols_scaled

X_train_final = pd.DataFrame(X_train_processed, columns=final_columns)
X_eval_final = pd.DataFrame(X_eval_processed, columns=final_columns)
X_test_final = pd.DataFrame(X_test_processed, columns=final_columns)

# Opcional: convertir a numérico (int/float) para evitar object
X_train_final = X_train_final.apply(pd.to_numeric)
X_eval_final = X_eval_final.apply(pd.to_numeric)
X_test_final = X_test_final.apply(pd.to_numeric)

# Guardar X procesados
X_train_final.to_parquet("data/processed/X_train.parquet", index=False)
X_eval_final.to_parquet("data/processed/X_eval.parquet", index=False)
X_test_final.to_parquet("data/processed/X_test.parquet", index=False)

# Guardar y separados (target)
y_train.to_frame().to_parquet("data/processed/y_train.parquet", index=False)
y_eval.to_frame().to_parquet("data/processed/y_eval.parquet", index=False)
y_test.to_frame().to_parquet("data/processed/y_test.parquet", index=False)
