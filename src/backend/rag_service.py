import httpx
import os
import json
from typing import List

class RAGExplanationService:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "your-api-key-placeholder")
        # Knowledge base derived from KDD data exploration and network security patterns
        self.knowledge_base = {
            "dos": {
                "description": "Denial of Service (DoS) attack.",
                "details": "This attack attempts to shut down a machine or network, making it inaccessible to its intended users. Common types include SYN Flooding (S0 flags), Smurf attacks (ICMP flood), and Teardrop attacks (fragmentation errors).",
                "indicators": ["High serror_rate", "Many same-destination packets in short window", "Large src_bytes"]
            },
            "probe": {
                "description": "Probing or Surveillance attack.",
                "details": "The attacker scans a network to gather information or find vulnerabilities. Examples include Nmap scans, IP sweeping, and port scanning. It often involves many connection attempts to different ports (various services).",
                "indicators": ["High diff_srv_rate", "Many different services to same host", "REJ flags"]
            },
            "r2l": {
                "description": "Remote to Local (R2L) attack.",
                "details": "An attacker sends packets to a machine over a network but does not have an account on that machine and exploits some vulnerability to gain local access as a user.",
                "indicators": ["Use of specific services like ftp, telnet, imap", "High number of failed logins", "Guest login attempts"]
            },
            "u2r": {
                "description": "User to Root (U2R) attack.",
                "details": "The attacker starts with access to a normal user account on the system and is able to exploit some vulnerability to gain root/administrator access.",
                "indicators": ["Use of shell/telnet services", "Attempts to gain root access (su_attempted)", "File creation in system directories"]
            },
            "general_anomaly": {
                "description": "Statistical Network Anomaly.",
                "details": "The hybrid model detected a reconstruction error or statistical outlier that deviates significantly from the 'normal' baseline traffic profile.",
                "indicators": ["High hybrid anomaly score", "Unusual combination of flags and byte counts"]
            }
        }

        # Prepare TF-IDF index for local semantic search
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            self.kb_texts = []
            self.kb_keys = []
            for key, val in self.knowledge_base.items():
                self.kb_keys.append(key)
                text = f"{val['description']} {val['details']} {' '.join(val['indicators'])}"
                self.kb_texts.append(text)
                
            self.vectorizer = TfidfVectorizer(stop_words='english')
            self.tfidf_matrix = self.vectorizer.fit_transform(self.kb_texts)
            self.use_tfidf = True
        except ImportError:
            self.use_tfidf = False

    async def explain_anomaly(self, packet_info: dict, hybrid_score: float):
        """
        Explains why a specific network behavior was flagged as an anomaly using the knowledge base.
        """
        attack_category = self._classify_anomaly(packet_info)
        context = self.knowledge_base.get(attack_category, self.knowledge_base["general_anomaly"])

        # Build base explanation
        explanation = f"**Classification:** {context['description']}\n\n"
        explanation += f"**Analysis:** {context['details']}\n\n"
        explanation += f"**Indicators Observed:** {', '.join(context['indicators'])}\n\n"
        explanation += f"**Technical Details:** Packet from {packet_info.get('src', 'unknown')} to {packet_info.get('dst', 'unknown')} using protocol {packet_info.get('proto', 'unknown')} (Service: {packet_info.get('service', 'unknown')}) was flagged with a high anomaly score of {hybrid_score:.4f}."

        # Perform local TF-IDF Search
        if self.use_tfidf:
            try:
                from sklearn.metrics.pairwise import cosine_similarity
                q_tokens: List[str] = []
                # create a short query string from observed fields
                for k in ('service', 'proto', 'src_bytes', 'dst_host_count', 'diff_srv_rate', 'num_failed_logins', 'serror_rate'):
                    if k in packet_info:
                        q_tokens.append(f"{k} {packet_info[k]}")
                q_tokens.append(attack_category)
                query_text = ' '.join(map(str, q_tokens))
                
                query_vec = self.vectorizer.transform([query_text])
                sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
                
                top_indices = sims.argsort()[-3:][::-1]
                
                hits_found = False
                hit_text = "\n\n**Knowledge Base Matches:**\n"
                for idx in top_indices:
                    score = sims[idx]
                    if score > 0.05: # Threshold
                        hits_found = True
                        k = self.kb_keys[idx]
                        kb_item = self.knowledge_base[k]
                        hit_text += f"- {kb_item['description']} [score={score:.3f}]\n"
                        
                if hits_found:
                    explanation += hit_text
            except Exception as e:
                print(f"TF-IDF search error: {e}")

        # If API key is available, we could enhance this with an LLM call.
        if self.api_key != "your-api-key-placeholder":
            try:
                # Placeholder for actual LLM integration if requested
                pass
            except Exception:
                pass
        
        return explanation

    def _classify_anomaly(self, info):
        # Heuristic classification based on the extracted features and KDD patterns
        proto = str(info.get('proto')).lower()
        service = info.get('service', '').lower()
        
        if info.get('serror_rate', 0) > 0.5 or info.get('src_bytes', 0) > 50000:
            return "dos"
        if service in ["telnet", "ftp", "imap4"]:
            return "r2l"
        if service in ["shell", "login"]:
            return "u2r"
        if info.get('diff_srv_rate', 0) > 0.5:
            return "probe"
            
        return "general_anomaly"
