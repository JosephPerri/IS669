import pandas as pd
import glob
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures  

# Get a list of CSV files in the specified directory
jira_case_files = glob.glob("./inputs/*.csv")

# Define the columns you want to select and analyze
feature_columns = ['Project name', 'Project lead', 'Priority', 'Assignee', 'Reporter', 'Creator', 'Component/s']

# Initialize a list to store DataFrames from each CSV file
df_jira_list = []

# Loop through each CSV file
for file in jira_case_files:
    # Read the CSV file into a DataFrame
    df_csv = pd.read_csv(file)
    
    # Replace empty strings with None values
    df_csv.replace("", None, inplace=True)

    # Convert date strings to datetime objects
    df_csv['Created'] = pd.to_datetime(df_csv['Created'])
    df_csv['Resolved'] = pd.to_datetime(df_csv['Resolved'])
    
    # Calculate the 'Days to resolution' by subtracting 'Created' from 'Resolved'
    df_csv['Days to resolution'] = (df_csv['Resolved'] - df_csv['Created']).dt.days
    
    # Select the desired columns for analysis
    selected_fields = feature_columns + ['Days to resolution']
    df_csv = df_csv[selected_fields]
    
    # Append the DataFrame to the list
    df_jira_list.append(df_csv) 

# Concatenate all DataFrames into a single DataFrame
df = pd.concat(df_jira_list)

# Get average day to resolution
average_day_to_resolution = df['Days to resolution'].mean()

# Initialize a label encoder for encoding categorical features
label_encoder = LabelEncoder()

# Apply label encoding to categorical columns in the DataFrame
df[feature_columns] = df[feature_columns].apply(lambda x: label_encoder.fit_transform(x))

# Split the data into features (X) and the target variable (y)
X = df.drop(columns=['Days to resolution'])
y = df['Days to resolution']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (optional but often beneficial for some models)
scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

# Initialize a Linear Regression model
lin = LinearRegression()

# Add polynomial features to the standardized training and testing data
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train_scaler)
X_test_poly = poly.transform(X_test_scaler)

# Fit the Linear Regression model on the polynomial features
poly.fit(X_poly_train, y_train)
lin.fit(X_poly_train, y_train)

# Make predictions on the test data
y_pred = lin.predict(X_test_poly)

# Calculate the mean absolute error on the test data
mae_test = mean_absolute_error(y_test, y_pred)

# Make predictions on the training data
y_pred_train = lin.predict(X_poly_train)

# Calculate the mean absolute error on the training data
mae_train = mean_absolute_error(y_train, y_pred_train)

# Print the average and mean absolutes errors
print("Average days to resolution", average_day_to_resolution)
print("Mean Absolute Error on Test Data:", mae_test)
print("Mean Absolute Error on Training Data:", mae_train)
