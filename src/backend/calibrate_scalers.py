import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.backend.detector import HybridDetector
from src.backend.sniffer import RealTimeFeatureExtractor

def main():
    model_dir = os.path.join(os.path.dirname(__file__), "saved_models")
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data/mock_dataset.csv")
    
    print(f"Loading detector and extractor...")
    detector = HybridDetector(model_dir)
    extractor = RealTimeFeatureExtractor(
        feature_columns_path=os.path.join(model_dir, "feature_columns.txt"),
        scaler_path=os.path.join(model_dir, "standard_scaler.joblib"),
        models_dir=model_dir
    )
    
    print(f"Loading calibration data from {data_path}...")
    # Load CSV - skip header if exists, handle appropriately
    # The mock_dataset.csv we created has 43 columns
    df_raw = pd.read_csv(data_path, header=None)
    
    mses = []
    if_scores = []
    
    print(f"Processing {len(df_raw)} rows for calibration...")
    for i, row in df_raw.iterrows():
        # Map row to features (mimicking sniffer logic)
        # Note: protocol_type (1), service (2), flag (3)
        data = {
            'duration': row[0], 'protocol_type': row[1], 'service': row[2], 'flag': row[3],
            'src_bytes': row[4], 'dst_bytes': row[5], 'land': row[6], 'wrong_fragment': row[7],
            'urgent': row[8], 'hot': row[9], 'num_failed_logins': row[10], 'logged_in': row[11],
            'num_compromised': row[12], 'root_shell': row[13], 'su_attempted': row[14],
            'num_root': row[15], 'num_file_creations': row[16], 'num_shells': row[17],
            'num_access_files': row[18], 'num_outbound_cmds': row[19], 'is_host_login': row[20],
            'is_guest_login': row[21], 'count': row[22], 'srv_count': row[23],
            'serror_rate': row[24], 'srv_serror_rate': row[25], 'rerror_rate': row[26],
            'srv_rerror_rate': row[27], 'same_srv_rate': row[28], 'diff_srv_rate': row[29],
            'srv_diff_host_rate': row[30], 'dst_host_count': row[31], 'dst_host_srv_count': row[32],
            'dst_host_same_srv_rate': row[33], 'dst_host_diff_srv_rate': row[34],
            'dst_host_same_src_port_rate': row[35], 'dst_host_srv_diff_host_rate': row[36],
            'dst_host_serror_rate': row[37], 'dst_host_srv_serror_rate': row[38],
            'dst_host_rerror_rate': row[39], 'dst_host_srv_rerror_rate': row[40]
        }
        
        df_feat = pd.DataFrame([data])
        # Encode categorical
        try: df_feat['protocol_type'] = extractor.le_proto.transform(df_feat['protocol_type'])
        except: df_feat['protocol_type'] = 0
        try: df_feat['service'] = extractor.le_service.transform(df_feat['service'])
        except: df_feat['service'] = 0
        try: df_feat['flag'] = extractor.le_flag.transform(df_feat['flag'])
        except: df_feat['flag'] = 0
        
        # Scale
        scaled = extractor.scaler.transform(df_feat.values)
        
        # Get raw scores
        reconstruction = detector.autoencoder.predict(scaled, verbose=0)
        mse = np.mean(np.power(scaled - reconstruction, 2), axis=1)
        if_score = -detector.iso_forest.decision_function(scaled)
        
        mses.append(mse[0])
        if_scores.append(if_score[0])

    print(f"Fitting scalers...")
    ae_scaler = MinMaxScaler().fit(np.array(mses).reshape(-1, 1))
    if_scaler = MinMaxScaler().fit(np.array(if_scores).reshape(-1, 1))
    
    print(f"Saving scalers to {model_dir}...")
    joblib.dump(ae_scaler, os.path.join(model_dir, "ae_scaler.joblib"))
    joblib.dump(if_scaler, os.path.join(model_dir, "if_scaler.joblib"))
    
    print("Calibration complete!")

if __name__ == "__main__":
    main()
