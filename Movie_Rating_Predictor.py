import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

Dataset = "IMDb Movies India.csv"

def load_data(file_path):
    # Loading  data with encoding
    return pd.read_csv(file_path, encoding="ISO-8859-1")

def preprocess_data(df):
    # Dropping rows with missing target values and creating a copy
    df = df.dropna(subset=['Rating']).copy()
    
    # Filling missing values for categorical columns
    for col in ['Director', 'Genre', 'Actor 1', 'Actor 2', 'Actor 3']:
        if col in df.columns:
            df.loc[:, col] = df[col].fillna('Unknown')
    
    # Cleaning Year column-> extract digits and convert to numeric
    if 'Year' in df.columns:
        df.loc[:, 'Year'] = df['Year'].astype(str).str.extract(r'(\d+)', expand=False)
        df.loc[:, 'Year'] = pd.to_numeric(df['Year'], errors='coerce')
    
    # Cleaning Duration column-> extract digits and convert to numeric
    if 'Duration' in df.columns:
        df.loc[:, 'Duration'] = df['Duration'].astype(str).str.extract(r'(\d+)', expand=False)
        df.loc[:, 'Duration'] = pd.to_numeric(df['Duration'], errors='coerce')
    
    # Process numeric columns: Votes and Metascore if present
    for col in ['Year', 'Duration', 'Votes', 'Metascore']:
        if col in df.columns:
            df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')
            median_val = df[col].median(skipna=True)
            df.loc[:, col] = df[col].fillna(median_val)
    
    # Ensuring Rating is numeric
    df.loc[:, 'Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    return df

def feature_engineering(df):
    # Computing director success rate
    df.loc[:, 'director_success_rate'] = df.groupby('Director')['Rating'].transform('mean')
    
    # Extracting primary genre from Genre column
    df.loc[:, 'Primary_Genre'] = df['Genre'].apply(lambda x: x.split(',')[0] if isinstance(x, str) else 'Unknown')
    
    # Computing genre average rating
    df.loc[:, 'genre_avg_rating'] = df.groupby('Primary_Genre')['Rating'].transform('mean')
    
    # Computing actor success rate for each actor column
    actor_cols = ['Actor 1', 'Actor 2', 'Actor 3']
    actor_avgs = {}
    for col in actor_cols:
        if col in df.columns:
            actor_avgs[col] = df.groupby(col)['Rating'].mean().to_dict()
    
    # Calculate overall actor success rate per row
    def calc_actor_success(row):
        rates = []
        for col in actor_cols:
            if col in row and row[col] != 'Unknown':
                avg = actor_avgs[col].get(row[col], np.nan)
                if not pd.isna(avg):
                    rates.append(avg)
        if rates:
            return np.mean(rates)
        else:
            return np.nan
    df.loc[:, 'actor_success_rate'] = df.apply(calc_actor_success, axis=1)
    overall_median = df['Rating'].median()
    df.loc[:, 'actor_success_rate'] = df['actor_success_rate'].fillna(overall_median)
    
    return df

def get_features(df):
    # List of features for training and prediction
    base_features = ['Year', 'Duration', 'Votes', 'Metascore', 
                     'director_success_rate', 'genre_avg_rating', 'actor_success_rate']
    return [col for col in base_features if col in df.columns]

def build_model(df):
    # Selecting features and target
    training_features = get_features(df)
    X = df[training_features]
    y = df['Rating']
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test, training_features

def evaluate_model(model, X_test, y_test):
 
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    accuracy = max(0, r2) * 100
    print("RMSE: {:.3f}".format(rmse))
    print("MAE: {:.3f}".format(mae))
    print("R² Score: {:.3f}".format(r2))
    print("Accuracy: {:.2f}%".format(accuracy))

def predict_new_movie(model, new_movie, training_features):
   
    sample_df = pd.DataFrame([new_movie])
    sample_df = preprocess_data(sample_df)
    sample_df = feature_engineering(sample_df)
    X_sample = sample_df.reindex(columns=training_features)
    predicted_rating = model.predict(X_sample)[0]
    print("Predicted Rating for '{}' : {:.2f}".format(new_movie.get('Title', 'New Movie'), predicted_rating))
    return predicted_rating

def main():
    df = load_data(Dataset)
    print("Dataset loaded successfully. Shape: {}".format(df.shape))
    df = preprocess_data(df)
    print("Data preprocessing completed.")
    df = feature_engineering(df)
    print("Feature engineering completed.")
    model, X_test, y_test, training_features = build_model(df)
    print("Model training completed.")
    evaluate_model(model, X_test, y_test)
    
    new_movie = {
        'Title': 'Baby John',
        'Director': 'Kalees',
        'Genre': 'Drama, Action',
        'Year': 2024,
        'Duration': '165 min',
        'Votes': 12500,
        'Metascore': 80,
        'Actor 1': 'Varun Dhawan',
        'Actor 2': 'Salman Khan',
        'Actor 3': 'Keerthy Suresh',
        'Rating': 0
    }
    
    print("\nPredicting for a new movie:")
    predict_new_movie(model, new_movie, training_features)

if __name__ == "__main__":
    main()
