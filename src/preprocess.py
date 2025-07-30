import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path


def preprocess_data(raw_data_path: str, processed_path: str):
    """Loads, preprocesses, and splits the Iris dataset."""
    Path(processed_path).mkdir(parents=True, exist_ok=True)

    # Load data, assuming the CSV is in the specified path
    iris_df = pd.read_csv(raw_data_path)

    # Simple preprocessing: Renaming columns for consistency
    iris_df.columns = ['Id', 'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    iris_df = iris_df.drop('Id', axis=1)

    # Split data
    train_df, test_df = train_test_split(iris_df, test_size=0.2, random_state=42, stratify=iris_df['species'])

    # Save processed data
    train_df.to_csv(Path(processed_path) / 'train.csv', index=False)
    test_df.to_csv(Path(processed_path) / 'test.csv', index=False)
    print(f"Data preprocessed and saved to {processed_path}")

if __name__ == '__main__':
    preprocess_data('data/raw/iris.csv', 'data/processed')