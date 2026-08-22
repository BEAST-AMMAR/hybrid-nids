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
        
        self.le_proto = joblib.load(os.path.join(models_dir, "le_proto.joblib"))
        self.le_service = joblib.load(os.path.join(models_dir, "le_service.joblib"))
        self.le_flag = joblib.load(os.path.join(models_dir, "le_flag.joblib"))
        
        self.window_size = 2.0
        self.host_history = {} # dst_ip -> list of (timestamp, service, is_error, src_ip, src_port)
        
        self.port_to_service = {
            20: "ftp_data", 21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp",
            42: "name", 53: "domain", 70: "gopher", 79: "finger", 80: "http",
            109: "pop_2", 110: "pop_3", 113: "auth", 119: "nntp", 135: "ntp_u",
            139: "netbios_ssn", 143: "imap4", 161: "snmp", 179: "bgp",
            389: "ldap", 443: "http_443", 512: "exec", 513: "login",
            514: "shell", 515: "printer", 540: "uucp", 6667: "IRC"
        }

    def _cleanup_history(self, current_time):
        """Remove packets older than window_size efficiently."""
        cutoff = current_time - self.window_size
        for dst in list(self.host_history.keys()):
            # Filter old packets
            self.host_history[dst] = [p for p in self.host_history[dst] if p[0] > cutoff]
            if not self.host_history[dst]:
                del self.host_history[dst]

    def packets_to_features(self, packets):
        """Process a batch of packets, update flow states, and return scaled features."""
        if not packets:
            return None, []
            
        current_time = time.time()
        self._cleanup_history(current_time)
        
        data_list = []
        valid_packet_infos = []
        
        for packet in packets:
            if not packet.haslayer(scapy.IP):
                continue
                
            ip_layer = packet[scapy.IP]
            src = ip_layer.src
            dst = ip_layer.dst
            
            protocol = "other"
            if packet.haslayer(scapy.TCP): protocol = "tcp"
            elif packet.haslayer(scapy.UDP): protocol = "udp"
            elif packet.haslayer(scapy.ICMP): protocol = "icmp"
            
            service = "other"
            sport, dport = 0, 0
            if packet.haslayer(scapy.TCP):
                sport, dport = packet[scapy.TCP].sport, packet[scapy.TCP].dport
            elif packet.haslayer(scapy.UDP):
                sport, dport = packet[scapy.UDP].sport, packet[scapy.UDP].dport
            service = self.port_to_service.get(sport, self.port_to_service.get(dport, "other"))
            
            flag = "SF"
            if packet.haslayer(scapy.TCP):
                flags_str = str(packet[scapy.TCP].flags)
                if flags_str == 'S': flag = "S0"
                elif flags_str == 'SA': flag = "S1"
                elif flags_str == 'REJ': flag = "REJ"
                elif 'R' in flags_str: flag = "RSTR" if 'A' in flags_str else "RSTO"
                else: flag = "SF"

            src_bytes = len(ip_layer.payload)
            dst_bytes = 0 
            land = 1 if src == dst else 0
            wrong_fragment = ip_layer.frag
            urgent = packet[scapy.TCP].urgptr if packet.haslayer(scapy.TCP) else 0
            
            is_error = 1 if flag in ["REJ", "RSTR", "RSTO", "S0"] else 0
            
            # Update efficient dictionary-based history
            if dst not in self.host_history:
                self.host_history[dst] = []
            self.host_history[dst].append((current_time, service, is_error, src, sport))
            
            # Calculate metrics for THIS destination
            hist = self.host_history[dst]
            count = len(hist)
            srv_count = sum(1 for p in hist if p[1] == service)
            serror_count = sum(1 for p in hist if p[2] == 1)
            srv_serror_count = sum(1 for p in hist if p[1] == service and p[2] == 1)
            same_src_port_count = sum(1 for p in hist if p[3] == src and p[4] == sport)
            
            serror_rate = serror_count / count if count > 0 else 0.0
            srv_serror_rate = srv_serror_count / srv_count if srv_count > 0 else 0.0
            same_srv_rate = srv_count / count if count > 0 else 1.0
            diff_srv_rate = 1.0 - same_srv_rate
            dst_host_same_src_port_rate = same_src_port_count / count if count > 0 else 0.0
            
            data = {
                'duration': 0, 'protocol_type': protocol, 'service': service, 'flag': flag,
                'src_bytes': src_bytes, 'dst_bytes': dst_bytes, 'land': land, 'wrong_fragment': wrong_fragment,
                'urgent': urgent, 'hot': 0, 'num_failed_logins': 0, 'logged_in': 0, 'num_compromised': 0,
                'root_shell': 0, 'su_attempted': 0, 'num_root': 0, 'num_file_creations': 0, 'num_shells': 0,
                'num_access_files': 0, 'num_outbound_cmds': 0, 'is_host_login': 0, 'is_guest_login': 0,
                'count': count, 'srv_count': srv_count, 'serror_rate': serror_rate, 'srv_serror_rate': srv_serror_rate,
                'rerror_rate': 0, 'srv_rerror_rate': 0, 'same_srv_rate': same_srv_rate, 'diff_srv_rate': diff_srv_rate,
                'srv_diff_host_rate': 0.0, 'dst_host_count': count, 'dst_host_srv_count': srv_count,
                'dst_host_same_srv_rate': same_srv_rate, 'dst_host_diff_srv_rate': diff_srv_rate,
                'dst_host_same_src_port_rate': dst_host_same_src_port_rate, 'dst_host_srv_diff_host_rate': 0.0,
                'dst_host_serror_rate': serror_rate, 'dst_host_srv_serror_rate': srv_serror_rate,
                'dst_host_rerror_rate': 0.0, 'dst_host_srv_rerror_rate': 0.0
            }
            data_list.append(data)
            
            packet_info = {
                "src": src,
                "dst": dst,
                "proto": protocol,
                "service": service,
                "size": len(packet)
            }
            valid_packet_infos.append(packet_info)

        if not data_list:
            return None, []
            
        df = pd.DataFrame(data_list)
        
        # Enforce column order to prevent scaling errors
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_columns]
        
        # Categorical Encoding
        try:
            df['protocol_type'] = self.le_proto.transform(df['protocol_type'])
        except: df['protocol_type'] = 0
        try:
            df['service'] = self.le_service.transform(df['service'])
        except: df['service'] = 0
        try:
            df['flag'] = self.le_flag.transform(df['flag'])
        except: df['flag'] = 0

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            scaled_features = self.scaler.transform(df.values)
            
        return scaled_features, valid_packet_infos

def start_sniffing(callback):
    print("Starting packet sniffing...")
    try:
        scapy.sniff(prn=callback, store=0)
    except PermissionError:
        print("Permission denied: Packet sniffing requires administrative privileges (sudo).")
    except Exception as e:
        print(f"Error during sniffing: {e}")
