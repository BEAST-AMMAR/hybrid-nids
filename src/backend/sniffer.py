import scapy.all as scapy
import pandas as pd
import numpy as np
import os
import joblib
import time

class RealTimeFeatureExtractor:
    def __init__(self, feature_columns_path, scaler_path, models_dir):
        with open(feature_columns_path, "r") as f:
            self.feature_columns = [line.strip() for line in f]
        self.scaler = joblib.load(scaler_path)

        # Label encoders for categorical features
        self.le_proto = joblib.load(os.path.join(models_dir, "le_proto.joblib"))
        self.le_service = joblib.load(os.path.join(models_dir, "le_service.joblib"))
        self.le_flag = joblib.load(os.path.join(models_dir, "le_flag.joblib"))

        # Window-based feature tracking
        self.packet_history = []

        # Mapping for common ports to services
        self.port_to_service = {
            20: "ftp_data", 21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp",
            42: "name", 53: "domain", 70: "gopher", 79: "finger", 80: "http",
            109: "pop_2", 110: "pop_3", 113: "auth", 119: "nntp", 135: "ntp_u",
            139: "netbios_ssn", 143: "imap4", 161: "snmp", 179: "bgp",
            389: "ldap", 443: "http_443", 512: "exec", 513: "login",
            514: "shell", 515: "printer", 540: "uucp", 6667: "IRC"
        }

    def packet_to_features(self, packet):
        if not (packet.haslayer(scapy.IP)):
            return None

        current_time = time.time()
        ip_layer = packet[scapy.IP]
        src = ip_layer.src
        dst = ip_layer.dst

        # 1. Basic features
        protocol = "other"
        if packet.haslayer(scapy.TCP): protocol = "tcp"
        elif packet.haslayer(scapy.UDP): protocol = "udp"
        elif packet.haslayer(scapy.ICMP): protocol = "icmp"

        # 2. Service Identification
        service = "other"
        sport = 0
        dport = 0
        if packet.haslayer(scapy.TCP):
            sport = packet[scapy.TCP].sport
            dport = packet[scapy.TCP].dport
        elif packet.haslayer(scapy.UDP):
            sport = packet[scapy.UDP].sport
            dport = packet[scapy.UDP].dport

        service = self.port_to_service.get(sport, self.port_to_service.get(dport, "other"))

        # 3. Flag Identification (TCP only)
        flag = "SF"
        if packet.haslayer(scapy.TCP):
            tcp_layer = packet[scapy.TCP]
            flags = tcp_layer.flags
            if flags == 'S': flag = "S0"
            elif flags == 'SA': flag = "S1"
            elif flags == 'REJ': flag = "REJ"
            elif 'R' in flags: flag = "RSTR" if 'A' in flags else "RSTO"
            else: flag = "SF"

        # 4. Intrinsic features
        src_bytes = len(ip_layer.payload)
        dst_bytes = 0
        land = 1 if ip_layer.src == ip_layer.dst else 0
        wrong_fragment = ip_layer.frag
        urgent = packet[scapy.TCP].urgptr if packet.haslayer(scapy.TCP) else 0

        # Update history for window-based features (2s window)
        is_error = 1 if flag in ["REJ", "RSTR", "RSTO", "S0"] else 0
        self.packet_history.append((current_time, src, dst, service, is_error))
        self.packet_history = [p for p in self.packet_history if current_time - p[0] <= 2.0]

        # Calculate counts
        count = sum(1 for p in self.packet_history if p[2] == dst)
        srv_count = sum(1 for p in self.packet_history if p[3] == service)
        serror_rate = sum(1 for p in self.packet_history if p[2] == dst and p[4] == 1) / count if count > 0 else 0
        srv_serror_rate = sum(1 for p in self.packet_history if p[3] == service and p[4] == 1) / srv_count if srv_count > 0 else 0

        # Create Dataframe
        data = {
            'duration': 0,
            'protocol_type': protocol,
            'service': service,
            'flag': flag,
            'src_bytes': src_bytes,
            'dst_bytes': dst_bytes,
            'land': land,
            'wrong_fragment': wrong_fragment,
            'urgent': urgent,
            'hot': 0, 'num_failed_logins': 0, 'logged_in': 0, 'num_compromised': 0, 'root_shell': 0, 'su_attempted': 0, 'num_root': 0, 'num_file_creations': 0, 'num_shells': 0, 'num_access_files': 0, 'num_outbound_cmds': 0, 'is_host_login': 0, 'is_guest_login': 0,
            'count': count,
            'srv_count': srv_count,
            'serror_rate': serror_rate,
            'srv_serror_rate': srv_serror_rate,
            'rerror_rate': 0, 'srv_rerror_rate': 0, 'same_srv_rate': 1.0, 'diff_srv_rate': 0.0, 'srv_diff_host_rate': 0.0,
            'dst_host_count': count, 'dst_host_srv_count': srv_count, 'dst_host_same_srv_rate': 1.0, 'dst_host_diff_srv_rate': 0.0, 'dst_host_same_src_port_rate': 1.0, 'dst_host_srv_diff_host_rate': 0.0, 'dst_host_serror_rate': serror_rate, 'dst_host_srv_serror_rate': srv_serror_rate, 'dst_host_rerror_rate': 0.0, 'dst_host_srv_rerror_rate': 0.0
        }

        df = pd.DataFrame([data])

        # Encode categorical
        try:
            df['protocol_type'] = self.le_proto.transform(df['protocol_type'])
        except: df['protocol_type'] = 0

        try:
            df['service'] = self.le_service.transform(df['service'])
        except: df['service'] = 0

        try:
            df['flag'] = self.le_flag.transform(df['flag'])
        except: df['flag'] = 0

        # Scale
        scaled_features = self.scaler.transform(df)
        return scaled_features

def start_sniffing(callback):
    print("Starting packet sniffing...")
    try:
        scapy.sniff(prn=callback, store=0)
    except PermissionError:
        print("Permission denied: Packet sniffing requires administrative privileges (sudo).")
    except Exception as e:
        print(f"Error during sniffing: {e}")
