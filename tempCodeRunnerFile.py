
    print("Feature engineering completed.")
    model, X_test, y_test, training_features = build_model(df)
    print("Model training completed.")
    evaluate_model(model, X_test, y_test)
    
    new_movie = {
        'Title': 'Future Blockbuster',
        'Director': 'John Doe',