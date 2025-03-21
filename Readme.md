Movie Rating Predictor
1. Task Objectives -->
→ Predict Movie Ratings:
• Build a model to estimate movie ratings using various attributes.

→ Data Preprocessing:
• Clean the raw dataset by handling missing values, converting data types, and extracting numeric information from text fields.

→ Feature Engineering:
• Create additional features such as:
• Director Success Rate: Average rating of movies for each director.
• Genre Average Rating: Average rating for movies within the primary genre.
• Actor Success Rate: Combined measure based on lead actors’ historical ratings.

→ Model Training and Evaluation:
• Train a Random Forest Regressor using the processed data.
• Evaluate performance using RMSE, MAE, R² Score, and an approximate accuracy metric.

→ New Movie Prediction:
• Implement functionality to predict the rating for a new, unseen movie using the same preprocessing and feature engineering steps.

2. Approach-->
a.Data Loading
→ Dataset:
• The dataset is read from IMDb Movies India.csv using the ISO‑8859‑1 encoding to handle special characters.

b.Data Preprocessing
→ Missing Target Removal:
• Rows without a Rating are dropped (as it is the target variable).

→ Handling Categorical Data:
• Missing values in columns such as Director, Genre, Actor 1, Actor 2, and Actor 3 are replaced with "Unknown".

→ Numeric Conversion and Cleaning:
• Year: Extract numeric digits from values (e.g., from (2019)) and convert to a numeric type.
• Duration: Extract the numeric part from values (e.g., "120 min") and convert to numeric.
• Votes & Metascore: Convert these columns to numeric values; fill missing values with the median.

c.Feature Engineering
→ Director Success Rate:
• Compute the average rating for movies directed by each director.

→ Primary Genre & Genre Average Rating:
• Extract the primary genre (the first genre listed) and calculate the average rating for that genre.

→ Actor Success Rate:
• Use data from Actor 1, Actor 2, and Actor 3 to compute an overall success rate for the lead actors.

3. Model Training and Evaluation
→ Model:
• A Random Forest Regressor is trained on an 80% training and 20% testing split.

→ Evaluation Metrics:
• RMSE: Root Mean Squared Error – measures average error magnitude.
• MAE: Mean Absolute Error – measures average absolute differences.
• R² Score: Indicates the proportion of variance explained by the model.
• Accuracy: Derived from R² (calculated as max(0, R²) * 100).

4. New Movie Prediction
→ Consistency:
• New movie data undergoes the same preprocessing and feature engineering as the training data.
→ Prediction:
• The trained model predicts the rating based on the processed input features.

