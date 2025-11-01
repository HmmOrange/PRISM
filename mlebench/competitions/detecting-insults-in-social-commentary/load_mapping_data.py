import os
import pandas as pd

def load_train_data(path: str):
    public_path = os.path.join(path, "prepared", "public")
    train_csv_path = os.path.join(public_path, "train.csv")
    train_df = pd.read_csv(train_csv_path)
    data = []
    for idx, row in train_df.iterrows():
        input_data = {
            "text_data": {
                "text": row["Comment"]
            },
            "label": {
                "Insult": row["Insult"]
            },
            "input_id": row["Comment"],
        }
        data.append(input_data)
    return data, pd.DataFrame()

def load_test_data(path: str):
    public_path = os.path.join(path, "prepared", "public")
    test_csv_path = os.path.join(public_path, "test.csv")
    test_df = pd.read_csv(test_csv_path)
    data = []
    for idx, row in test_df.iterrows():
        input_data = {
            "text_data": {
                "text": row["Comment"]
            },
            "input_id": idx,
            "fields": row.to_dict(),
            "predict_keys": ["Insult"]
        }
        data.append(input_data)
    return data, pd.DataFrame()

def load_lite_test_data(path: str):
    public_path = os.path.join(path, "prepared", "public")
    private_lite_path = os.path.join(path, "prepared_lite", "private")
    test_csv_path = os.path.join(public_path, "test.csv")
    test_label_path = os.path.join(private_lite_path, "answers.csv")
    
    test_df = pd.read_csv(test_csv_path)
    test_label_df = pd.read_csv(test_label_path)
    
    print(f"Debug info:")
    print(f"  Full test set: {len(test_df)} samples")
    print(f"  Lite test labels: {len(test_label_df)} samples")
    
    # Create composite key (Date, Comment) for unique identification
    test_df['composite_key'] = test_df['Date'].astype(str) + '|' + test_df['Comment'].astype(str)
    test_label_df['composite_key'] = test_label_df['Date'].astype(str) + '|' + test_label_df['Comment'].astype(str)
    
    # Filter test data to only include samples that are in the lite test set
    lite_keys = set(test_label_df["composite_key"].values)
    filtered_test_df = test_df[test_df["composite_key"].isin(lite_keys)]
    
    print(f"  Filtered test set: {len(filtered_test_df)} samples")
    print(f"  Match rate: {len(filtered_test_df)}/{len(test_label_df)} = {len(filtered_test_df)/len(test_label_df)*100:.1f}%")
    
    # Remove composite key before creating data
    filtered_test_df = filtered_test_df.drop(columns=['composite_key'])
    
    data = []
    for idx, row in filtered_test_df.iterrows():
        input_data = {
            "text_data": {
                "text": row["Comment"]
            },
            "input_id": idx,
            "fields": row.to_dict(),
            "predict_keys": ["Insult"]
        }
        data.append(input_data)
    
    # Remove composite key from test_label_df before returning
    test_label_df = test_label_df.drop(columns=['composite_key'])
    
    return data, test_label_df

def mapping(predictions: pd.DataFrame, path: str):
    submissions = predictions
    return submissions
