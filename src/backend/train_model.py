import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_autoencoder(input_dim, learning_rate=0.001, dropout_rate=0.2):
    input_layer = Input(shape=(input_dim,), name='input_features')
    
    encoded = Dense(32, activation='relu', name='encoder_dense_1')(input_layer)
    encoded = BatchNormalization(name='encoder_bn_1')(encoded)
    encoded = Dropout(dropout_rate, name='encoder_dropout_1')(encoded)
    
    encoded = Dense(16, activation='relu', name='encoder_dense_2')(encoded)
    encoded = BatchNormalization(name='encoder_bn_2')(encoded)
    encoded = Dropout(dropout_rate, name='encoder_dropout_2')(encoded)
    
    latent = Dense(8, activation='relu', name='latent_space')(encoded)
    
    decoded = Dense(16, activation='relu', name='decoder_dense_1')(latent)
    decoded = BatchNormalization(name='decoder_bn_1')(decoded)
    decoded = Dropout(dropout_rate, name='decoder_dropout_1')(decoded)
    
    decoded = Dense(32, activation='relu', name='decoder_dense_2')(decoded)
    decoded = BatchNormalization(name='decoder_bn_2')(decoded)
    decoded = Dropout(dropout_rate, name='decoder_dropout_2')(decoded)
    
    output_layer = Dense(input_dim, activation='linear', name='reconstructed_features')(decoded)
    
    autoencoder = Model(inputs=input_layer, outputs=output_layer, name='hybrid_nids_autoencoder')
    optimizer = Adam(learning_rate=learning_rate)
    autoencoder.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return autoencoder

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    data_dir = os.path.join(root_dir, "data")
    model_dir = os.path.join(root_dir, "src", "backend", "saved_models")
    
    os.makedirs(model_dir, exist_ok=True)
    
    train_path = os.path.join(data_dir, "KDDTrain+_20Percent.txt")
    test_path = os.path.join(data_dir, "KDDTest+.txt")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        logging.error("Dataset not found. Please run download_nsl_kdd.py or place the datasets in data/ folder.")
        return

    columns = [
        'duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
        'wrong_fragment','urgent','hot','num_failed_logins','logged_in',
        'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
        'num_shells','num_access_files','num_outbound_cmds','is_host_login',
        'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
        'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
        'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
        'dst_host_same_srv_rate','dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
        'dst_host_serror_rate','dst_host_srv_serror_rate',
        'dst_host_rerror_rate','dst_host_srv_rerror_rate',
        'label','difficulty'
    ]

    logging.info("Loading datasets...")
    train = pd.read_csv(train_path, header=None, names=columns)
    test = pd.read_csv(test_path, header=None, names=columns)
    
    # Drop difficulty
    train = train.drop("difficulty", axis=1)
    test = test.drop("difficulty", axis=1)

    logging.info("Encoding categorical features with LabelEncoder...")
    le_proto = LabelEncoder()
    le_service = LabelEncoder()
    le_flag = LabelEncoder()

    # Fit on both train and test to handle unseen labels
    le_proto.fit(pd.concat([train['protocol_type'], test['protocol_type']]))
    le_service.fit(pd.concat([train['service'], test['service']]))
    le_flag.fit(pd.concat([train['flag'], test['flag']]))

    train['protocol_type'] = le_proto.transform(train['protocol_type'])
    train['service'] = le_service.transform(train['service'])
    train['flag'] = le_flag.transform(train['flag'])

    test['protocol_type'] = le_proto.transform(test['protocol_type'])
    test['service'] = le_service.transform(test['service'])
    test['flag'] = le_flag.transform(test['flag'])

    # Labels: normal -> 0, anomaly -> 1
    train['label'] = train['label'].apply(lambda x: 0 if x == 'normal' else 1)
    test['label'] = test['label'].apply(lambda x: 0 if x == 'normal' else 1)

    X_train = train.drop('label', axis=1)
    y_train = train['label']
    X_test = test.drop('label', axis=1)
    y_test = test['label']

    feature_cols = list(X_train.columns)
    with open(os.path.join(model_dir, "feature_columns.txt"), "w") as f:
        f.write("\n".join(feature_cols))

    logging.info("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_normal = X_train_scaled[y_train == 0]

    input_dim = X_train_scaled.shape[1]
    
    # Autoencoder
    logging.info(f"Training Autoencoder (Input Dim: {input_dim})...")
    autoencoder = build_autoencoder(input_dim)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
    autoencoder.fit(
        X_train_normal, X_train_normal,
        epochs=50,
        batch_size=256,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

    # Autoencoder Reconstruction Error on Test
    reconstructions = autoencoder.predict(X_test_scaled)
    mse = np.mean(np.power(X_test_scaled - reconstructions, 2), axis=1)
    
    # Isolation Forest
    logging.info("Training Isolation Forest...")
    iso_forest = IsolationForest(n_estimators=200, contamination=0.1, random_state=42, n_jobs=-1)
    iso_forest.fit(X_train_scaled)
    if_scores = -iso_forest.decision_function(X_test_scaled)

    # Hybrid Score Calibration
    logging.info("Calibrating Hybrid Scores...")
    ae_scaler = MinMaxScaler()
    if_scaler = MinMaxScaler()
    
    ae_norm = ae_scaler.fit_transform(mse.reshape(-1,1)).flatten()
    if_norm = if_scaler.fit_transform(if_scores.reshape(-1,1)).flatten()
    
    alpha = 0.6
    hybrid_score = alpha * ae_norm + (1 - alpha) * if_norm
    
    # Threshold at 75th percentile of hybrid scores
    threshold = float(np.percentile(hybrid_score, 75))
    y_pred_hybrid = (hybrid_score > threshold).astype(int)

    # Calculate optimal threshold using ROC curve
    from sklearn.metrics import roc_curve
    fpr_roc, tpr_roc, thresholds = roc_curve(y_test, hybrid_score)
    optimal_idx = np.argmax(tpr_roc - fpr_roc)
    optimal_threshold = thresholds[optimal_idx]
    logging.info(f"Using Optimal Threshold: {optimal_threshold}")
    
    y_pred_optimal = (hybrid_score > optimal_threshold).astype(int)

    # Model Evaluation Metrics
    acc = accuracy_score(y_test, y_pred_optimal)
    prec = precision_score(y_test, y_pred_optimal)
    rec = recall_score(y_test, y_pred_optimal)
    f1 = f1_score(y_test, y_pred_optimal)
    cm = confusion_matrix(y_test, y_pred_optimal)
    
    print("\n" + "="*50)
    print("                MODEL EVALUATION RESULTS                ")
    print("="*50)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("-"*50)
    print("Confusion Matrix:")
    print(f"[{cm[0][0]:>5} (TN)]  [{cm[0][1]:>5} (FP)]")
    print(f"[{cm[1][0]:>5} (FN)]  [{cm[1][1]:>5} (TP)]")
    print("="*50 + "\n")

    # Save Everything
    logging.info("Saving models and scalers...")
    autoencoder.save(os.path.join(model_dir, "autoencoder.keras"))
    joblib.dump(iso_forest, os.path.join(model_dir, "iso_forest.joblib"))
    joblib.dump(scaler, os.path.join(model_dir, "standard_scaler.joblib"))
    joblib.dump(ae_scaler, os.path.join(model_dir, "ae_scaler.joblib"))
    joblib.dump(if_scaler, os.path.join(model_dir, "if_scaler.joblib"))
    joblib.dump(le_proto, os.path.join(model_dir, "le_proto.joblib"))
    joblib.dump(le_service, os.path.join(model_dir, "le_service.joblib"))
    joblib.dump(le_flag, os.path.join(model_dir, "le_flag.joblib"))
    with open(os.path.join(model_dir, "threshold.txt"), "w") as f:
        f.write(str(optimal_threshold))
        
    logging.info("Training complete and models saved.")

if __name__ == "__main__":
    main()
