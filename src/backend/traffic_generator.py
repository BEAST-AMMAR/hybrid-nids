import time
import random
import threading
import scapy.all as scapy
import queue
import logging

class SyntheticTrafficGenerator:
    def __init__(self):
        self.is_running = False
        self.thread = None
        self.virtual = True
        self.packet_queue = None

    def _generate_normal_packet(self):
        """Generates a standard, non-anomalous TCP/IP packet"""
        src_ip = f"192.168.1.{random.randint(2, 254)}"
        dst_ip = f"192.168.1.{random.randint(2, 254)}"
        sport = random.randint(1024, 65535)
        dport = random.choice([80, 443, 22, 53])
        
        # Standard HTTP/HTTPS request size
        payload = b"A" * random.randint(40, 1500)
        
        # Flags: mostly ACK or PSH+ACK
        flags = random.choice(['A', 'PA'])
        
        pkt = scapy.IP(src=src_ip, dst=dst_ip) / scapy.TCP(sport=sport, dport=dport, flags=flags) / scapy.Raw(load=payload)
        return pkt

    def _generate_anomaly_packet(self):
        """Generates anomalous packets that trigger ML features like 'land' or high 'count'"""
        anomaly_type = random.choice(['syn_flood', 'land_attack', 'port_scan'])
        
        if anomaly_type == 'syn_flood':
            # SYN flood: Generate a BURST of packets to trigger high `count` and `serror_rate`
            pkts = []
            dst_ip = "192.168.1.100"
            for _ in range(50):
                src_ip = f"{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}"
                pkts.append(scapy.IP(src=src_ip, dst=dst_ip) / scapy.TCP(sport=random.randint(1024, 65535), dport=80, flags='S'))
            return pkts
            
        elif anomaly_type == 'land_attack':
            # LAND attack: source IP == destination IP, which sets `land=1` in sniffer
            ip = "192.168.1.100"
            return [scapy.IP(src=ip, dst=ip) / scapy.TCP(sport=80, dport=80, flags='S')]
            
        else: # port_scan
            src_ip = "192.168.1.50" 
            dst_ip = "192.168.1.100"
            return [scapy.IP(src=src_ip, dst=dst_ip) / scapy.TCP(sport=54321, dport=random.randint(1, 65535), flags='S')]

    def _injection_loop(self):
        print(f"Started Synthetic Generator (Virtual={self.virtual})")
        while self.is_running:
            # 80% normal, 20% anomaly
            is_anomaly = random.random() < 0.2
            
            if is_anomaly:
                pkts = self._generate_anomaly_packet()
            else:
                pkts = [self._generate_normal_packet()]
                
            for pkt in pkts:
                if self.virtual and self.packet_queue is not None:
                    try:
                        self.packet_queue.put_nowait(pkt)
                    except queue.Full:
                        pass
                elif not self.virtual:
                    try:
                        scapy.send(pkt, verbose=False)
                    except Exception as e:
                        print(f"Failed to inject packet physically: {e}")
                    
            # Sleep slightly to simulate network delay
            time.sleep(random.uniform(0.02, 0.1))

    def start(self, virtual=True, target_queue=None):
        if self.is_running:
            return
        self.virtual = virtual
        self.packet_queue = target_queue
        self.is_running = True
        self.thread = threading.Thread(target=self._injection_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)

generator_instance = SyntheticTrafficGenerator()
