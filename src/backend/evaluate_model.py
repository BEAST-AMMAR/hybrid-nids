import sys
import os
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.backend.detector import HybridDetector
from src.backend.sniffer import RealTimeFeatureExtractor
from src.backend.traffic_generator import generator_instance
import scapy.all as scapy

def main():
    print("=======================================")
    print("      HYBRID NIDS MODEL EVALUATION     ")
    print("=======================================\n")
    
    model_dir = os.path.join(os.path.dirname(__file__), "saved_models")
    
    print("[1] Loading HybridDetector (Autoencoder + Isolation Forest)...")
    try:
        detector = HybridDetector(model_dir)
        print(f"    Loaded! Threshold set to: {detector.threshold:.4f}\n")
    except Exception as e:
        print(f"Failed to load detector: {e}")
        return

    print("[2] Loading Feature Extractor & Scalers...")
    try:
        extractor = RealTimeFeatureExtractor(
            feature_columns_path=os.path.join(model_dir, "feature_columns.txt"),
            scaler_path=os.path.join(model_dir, "standard_scaler.joblib"),
            models_dir=model_dir
        )
        print("    Loaded! Ready for real-time feature extraction.\n")
    except Exception as e:
        print(f"Failed to load extractor: {e}")
        return

    print("[3] Testing NORMAL traffic simulation...")
    # Generate some normal packets to build history
    print("    Generating 10 normal packets...")
    normal_pkts = [generator_instance._generate_normal_packet() for _ in range(10)]
    for pkt in normal_pkts:
        scaled_features = extractor.packet_to_features(pkt)
        time.sleep(0.01) # Small delay for window accumulation
    
    if scaled_features is not None:
        is_anomaly, score = detector.predict(scaled_features)
        print(f"    NORMAL TRAFFIC RESULT:")
        print(f"    -> Anomaly Score: {score:.4f} (Threshold: {detector.threshold:.4f})")
        print(f"    -> Classification: {'ANOMALY' if is_anomaly else 'NORMAL'}")
        if not is_anomaly:
            print("    -> PASS: Correctly classified as normal.\n")
        else:
            print("    -> FAIL: False positive.\n")
    else:
        print("    -> Failed to extract features.\n")
        
    print("[4] Testing ANOMALY traffic simulation (SYN FLOOD burst)...")
    print("    Generating 50 SYN packets...")
    # Force syn_flood for testing
    import random
    random.seed(42) # Try to force a specific outcome if needed, but we can just override anomaly_type
    
    # We will manually generate a burst of SYN packets from a random IP to a target IP
    syn_pkts = []
    dst_ip = "192.168.1.100"
    src_ip = "10.0.0.55"
    for _ in range(50):
        syn_pkts.append(scapy.IP(src=src_ip, dst=dst_ip) / scapy.TCP(sport=random.randint(1024, 65535), dport=80, flags='S'))
        
    for pkt in syn_pkts:
        scaled_features = extractor.packet_to_features(pkt)
        
    if scaled_features is not None:
        is_anomaly, score = detector.predict(scaled_features)
        print(f"    ANOMALY TRAFFIC RESULT:")
        print(f"    -> Anomaly Score: {score:.4f} (Threshold: {detector.threshold:.4f})")
        print(f"    -> Classification: {'ANOMALY' if is_anomaly else 'NORMAL'}")
        if is_anomaly:
            print("    -> PASS: Correctly classified as anomaly.\n")
        else:
            print("    -> FAIL: False negative.\n")
            
    print("[5] Testing ANOMALY traffic simulation (LAND ATTACK)...")
    print("    Generating LAND packet (src=dst)...")
    land_pkt = scapy.IP(src="192.168.1.100", dst="192.168.1.100") / scapy.TCP(sport=80, dport=80, flags='S')
    scaled_features = extractor.packet_to_features(land_pkt)
    if scaled_features is not None:
        is_anomaly, score = detector.predict(scaled_features)
        print(f"    LAND ATTACK RESULT:")
        print(f"    -> Anomaly Score: {score:.4f} (Threshold: {detector.threshold:.4f})")
        print(f"    -> Classification: {'ANOMALY' if is_anomaly else 'NORMAL'}")
        if is_anomaly:
            print("    -> PASS: Correctly classified as anomaly.\n")
        else:
            print("    -> FAIL: False negative.\n")

    print("=======================================")
    print("              DONE                     ")
    print("=======================================")

if __name__ == "__main__":
    main()
