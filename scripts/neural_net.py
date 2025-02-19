from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, cohen_kappa_score, log_loss, brier_score_loss, confusion_matrix, average_precision_score, fbeta_score
import numpy as np
import pandas as pd
import os

neuron=80

class PrintEpochCallback(Callback):
    def __init__(self, file):
        super().__init__()
        self.file = file
    
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            custom_print(f"Epoch {epoch+1}: {logs}", self.file)

"""Dataset specific preproc"""
def preprocess_data(df: pd.DataFrame):
    # column_drop_names = ["date", "time"]
    # Only graph
    column_drop_names = ["date", "time", "Updates", "A-Updates", "W-Updates", "A-Prefix", "W-Prefix", "A-Dup", "W-Dup", "AW-Mix", "Average AS-path length", "Maximum AS-path length", "Average packet size"]
    # Only stat
    # column_drop_names = ["date", "time", "max_betweenness_centrality","avg_betweenness_centrality","max_load_centrality","avg_load_centrality","max_closeness_centrality","avg_closeness_centrality","max_harmonic_centrality","avg_harmonic_centrality","max_degree_centrality","avg_degree_centrality","max_eigenvector_centrality","avg_eigenvector_centrality","max_number_of_cliques","avg_number_of_cliques","max_size_of_cliques","avg_size_of_cliques","max_clustering_coefficent","avg_clustering_coefficent","max_traingle_coefficient","avg_traingle_coefficient","max_square_clustering_coefficient","avg_square_clustering_coefficient","max_avg_neighbour_degree","avg_avg_neighbour_degree","max_eccentricity","avg_eccentricity"]
    columns_to_drop = [col for col in column_drop_names if col in df.columns]
    df = df.drop(columns_to_drop, axis=1)
    return df

"""Function to split the data into X and y based on target column"""
def get_x_y(df: pd.DataFrame, target_column="is_anamoly"):
    all_cols = df.columns.to_list()
    all_cols.remove(target_column)

    y = df[target_column]    
    X = df[all_cols]

    return X, y

def check_for_nans(data, f):
    for index, row in data.iterrows():
        nan_columns = row[row.isna()].index.tolist()
        if nan_columns:
            custom_print(f"Row {index} has NaN in columns: {', '.join(nan_columns)}", f)
            
    nan_rows = data[data.isna().any(axis=1)]
    custom_print(f"Row numbers with NaN values: {nan_rows.index.tolist()}", f)

def initial_checks(data, f):
    custom_print(f"Initial check for NaN values: {data.isnull().values.any()}", f)
    custom_print(f"Initial check for Infinite values: {np.isinf(data.values).any()}", f)

def standardize_data(X, f):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    custom_print(f"After scaling, NaN values in X: {np.isnan(X_scaled).any()}", f)
    custom_print(f"After scaling, Infinite values in X: {np.isinf(X_scaled).any()}", f)
    return X_scaled

def augment_data_with_noise(X, noise_level=0.02, times=2):
    augmented_data = X
    for _ in range(times):
        noise = np.random.normal(0, noise_level, X.shape)
        X_augmented = X + noise
        augmented_data = np.vstack((augmented_data, X_augmented))
    return augmented_data

def define_and_compile_model(input_shape, learning_rate=0.001):
    model = Sequential([
        Dense(neuron, activation='relu', input_shape=(input_shape,)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, file=None):
    callbacks = [PrintEpochCallback(file)]
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, verbose=1, callbacks=callbacks)
    return history

def evaluate_model(model, X_test, y_test, f):
    predictions_prob = model.predict(X_test)
    predictions = (predictions_prob > 0.5).astype(int).flatten()
    
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    average_precision = average_precision_score(y_test, predictions_prob.flatten())
    mcc = matthews_corrcoef(y_test, predictions)
    kappa = cohen_kappa_score(y_test, predictions)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ll = log_loss(y_test, predictions_prob.flatten())
    bs = brier_score_loss(y_test, predictions_prob.flatten())
    gmean = np.sqrt(sensitivity * specificity)
    csi = tp / (tp + fn + fp)
    f2_score = fbeta_score(y_test, predictions, beta=2)
    
    custom_print(f"Accuracy: {accuracy}", f)
    custom_print(f"Precision: {precision}", f)
    custom_print(f"Recall: {recall}", f)
    custom_print(f"F1 Score: {f1}", f)
    custom_print("Confusion Matrix:\n" + str(cm), f)
    custom_print("Average Precision-Recall Score: " + str(average_precision), f)
    custom_print("Matthews Correlation Coefficient: " + str(mcc), f)
    custom_print("Cohen’s Kappa: " + str(kappa), f)
    custom_print("Sensitivity (Recall): " + str(sensitivity), f)
    custom_print("Specificity: " + str(specificity), f)
    custom_print("Log Loss: " + str(ll), f)
    custom_print("Brier Score: " + str(bs), f)
    custom_print("G-Mean: " + str(gmean), f)
    custom_print("Critical Success Index (CSI): " + str(csi), f)
    custom_print("F-beta Score (beta=2): " + str(f2_score), f)

def custom_print(text, file=None):
    print(text)
    if file is not None:
        file.write(text + '\n')

if __name__ == "__main__":
    # Define the output file path
    output_file_path = f'/Users/yuvrajpatadia/Downloads/Network Routing/AnamolyDetection/scripts/data_proc/final_stats/single_{neuron}_neuron.txt'
    
    # Check if the directory exists, if not, create it
    output_directory = os.path.dirname(output_file_path)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # List of your files
    file_paths = [
        '/Users/yuvrajpatadia/Downloads/Network Routing/AnamolyDetection/scripts/data_proc/final_csvs/type1_labelwise_nans_avg_labelled_merged.csv',
        '/Users/yuvrajpatadia/Downloads/Network Routing/AnamolyDetection/scripts/data_proc/final_csvs/type2_adj_2w_pad_clean_merged.csv',
        '/Users/yuvrajpatadia/Downloads/Network Routing/AnamolyDetection/scripts/data_proc/final_csvs/type3_oneD_2W_df_clean_merged.csv',
        '/Users/yuvrajpatadia/Downloads/Network Routing/AnamolyDetection/scripts/data_proc/final_csvs/type4_twoD_2W_df_clean_merged.csv',
        '/Users/yuvrajpatadia/Downloads/Network Routing/AnamolyDetection/scripts/data_proc/final_csvs/type5_oneD_2W_df_clean_merged_v2.csv',
    ]
        
    with open(output_file_path, 'w') as f:
        for file_path in file_paths:
            # Print and write the name of the current file
            custom_print(f"Processing file: {file_path}", f)

            # Load and preprocess data
            data = pd.read_csv(file_path)
            data = preprocess_data(data)
            # print(data)
            check_for_nans(data, f)
            initial_checks(data, f)

            X, y = get_x_y(data, "is_anamoly")
       
            X_scaled = standardize_data(X, f)
            X_combined = augment_data_with_noise(X_scaled)
            y_combined = np.tile(y, 3)  # Replicate for augmented data
            
            # Split data
            X_train, X_temp, y_train, y_temp = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42, shuffle=True)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True)
            
            # Define, compile and train the model
            model = define_and_compile_model(X_train.shape[1])
            history = train_model(model, X_train, y_train, X_val, y_val, file=f)
            
            # Evaluate the model
            evaluate_model(model, X_test, y_test, f)

            custom_print("\n", f)

    print(f"Model evaluation has been saved to {output_file_path}")
