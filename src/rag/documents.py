"""
src/rag/documents.py
====================
All knowledge-base source documents for the NIDS RAG assistant.

Documents are returned as a list of dicts:
  {"id": str, "text": str, "metadata": dict}

Sources:
  1. NSL-KDD attack type documents (from config/mitre_mapping.json)
  2. MITRE ATT&CK tactic/technique summaries
  3. Network security concept guides
  4. Model explainability documentation
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from src.utils import get_project_root


# ------------------------------------------------------------------
# 1. NSL-KDD Attack Type Documents
# ------------------------------------------------------------------
def get_attack_documents() -> List[dict]:
    """
    Load attack type documents from config/mitre_mapping.json and
    return a rich text document for each attack.
    """
    mapping_path = get_project_root() / "config" / "mitre_mapping.json"
    with open(mapping_path, "r") as f:
        mapping = json.load(f)

    docs = []

    # Normal traffic
    normal = mapping.get("normal", {})
    docs.append({
        "id": "attack_normal",
        "text": (
            "Label: normal\n"
            "Category: Normal Traffic\n"
            "Description: Legitimate network traffic with no malicious intent. "
            "Normal connections follow expected communication patterns. "
            "The NIDS should classify these as benign.\n"
            "Response: No action required. Continue monitoring baseline traffic patterns."
        ),
        "metadata": {"type": "attack_doc", "attack": "normal", "category": "normal"},
    })

    for category, cat_data in mapping.get("categories", {}).items():
        cat_desc = cat_data.get("description", "")
        tactic   = cat_data.get("mitre_tactic", "")
        tactic_id = cat_data.get("mitre_tactic_id", "")

        for attack_name, attack in cat_data.get("attacks", {}).items():
            indicators_str  = ", ".join(attack.get("indicators",  []))
            response_str    = " | ".join(attack.get("response",   []))
            sub = attack.get("subtechnique") or ""
            sub_id = attack.get("subtechnique_id") or ""

            text = (
                f"Label: {attack_name}\n"
                f"Category: {category}\n"
                f"Severity: {attack.get('severity', 'unknown')}\n"
                f"Description: {attack.get('description', '')}\n"
                f"Category overview: {cat_desc}\n"
                f"MITRE ATT&CK Tactic: {tactic} ({tactic_id})\n"
                f"MITRE ATT&CK Technique: {attack.get('mitre_technique', '')} "
                f"({attack.get('mitre_technique_id', '')})\n"
            )
            if sub:
                text += f"MITRE Sub-technique: {sub} ({sub_id})\n"
            text += (
                f"Key Network Indicators: {indicators_str}\n"
                f"Recommended Response: {response_str}\n"
            )

            docs.append({
                "id": f"attack_{attack_name}",
                "text": text,
                "metadata": {
                    "type": "attack_doc",
                    "attack": attack_name,
                    "category": category,
                    "severity": attack.get("severity", "unknown"),
                    "mitre_technique_id": attack.get("mitre_technique_id", ""),
                },
            })

    return docs


# ------------------------------------------------------------------
# 2. MITRE ATT&CK Tactic Summaries
# ------------------------------------------------------------------
def get_mitre_tactic_documents() -> List[dict]:
    return [
        {
            "id": "mitre_recon",
            "text": (
                "MITRE ATT&CK Tactic: Reconnaissance (TA0043)\n"
                "Adversaries gather information to plan future operations. "
                "This includes active scanning (T1595), gathering host and network "
                "information, and identifying weaknesses. In the NSL-KDD dataset, "
                "Probe attacks (ipsweep, nmap, portsweep, satan, mscan, saint) "
                "correspond to this tactic. Key indicators: high scan rates, diverse "
                "destination hosts/ports, low bytes per connection, many SYN/reset flags."
            ),
            "metadata": {"type": "mitre_tactic", "tactic_id": "TA0043"},
        },
        {
            "id": "mitre_initial_access",
            "text": (
                "MITRE ATT&CK Tactic: Initial Access (TA0001)\n"
                "Adversaries gain entry to a network. Techniques include exploiting "
                "public-facing applications (T1190), phishing, and valid accounts (T1078). "
                "R2L attacks in NSL-KDD (guess_passwd, ftp_write, imap, sendmail, named, "
                "httptunnel, phf, warezmaster) map to this tactic. Indicators: failed "
                "login attempts, unusual service access, unexpected privilege escalation."
            ),
            "metadata": {"type": "mitre_tactic", "tactic_id": "TA0001"},
        },
        {
            "id": "mitre_privilege_esc",
            "text": (
                "MITRE ATT&CK Tactic: Privilege Escalation (TA0004)\n"
                "Adversaries gain higher-level permissions. Techniques include exploiting "
                "vulnerabilities (T1068), rootkits (T1014), and hijacking execution flow "
                "(T1574). U2R attacks in NSL-KDD (buffer_overflow, loadmodule, perl, "
                "rootkit, sqlattack, xterm, ps) map here. Indicators: root_shell=1, "
                "su_attempted=1, num_root > 0, unexpected kernel module loads."
            ),
            "metadata": {"type": "mitre_tactic", "tactic_id": "TA0004"},
        },
        {
            "id": "mitre_impact",
            "text": (
                "MITRE ATT&CK Tactic: Impact (TA0040)\n"
                "Adversaries disrupt availability or compromise integrity. Techniques "
                "include Network DoS (T1498) and Endpoint DoS (T1499). NSL-KDD DoS attacks "
                "(neptune, smurf, back, land, pod, teardrop, apache2, udpstorm, "
                "processtable, mailbomb) map here. Indicators: very high connection counts, "
                "high error rates, high bytes per connection, service unavailability."
            ),
            "metadata": {"type": "mitre_tactic", "tactic_id": "TA0040"},
        },
        {
            "id": "mitre_lateral",
            "text": (
                "MITRE ATT&CK Tactic: Lateral Movement (TA0008)\n"
                "Adversaries pivot through the network after initial access. Techniques "
                "include remote services (T1021) and exploitation (T1210). The NSL-KDD "
                "multihop R2L attack corresponds to this, where an attacker uses "
                "compromised intermediaries to reach the final target. Indicators: "
                "num_compromised > 0, is_guest_login=1, chained access patterns."
            ),
            "metadata": {"type": "mitre_tactic", "tactic_id": "TA0008"},
        },
    ]


# ------------------------------------------------------------------
# 3. Network Security Concept Guides
# ------------------------------------------------------------------
def get_security_concept_documents() -> List[dict]:
    return [
        {
            "id": "concept_ids_overview",
            "text": (
                "Network Intrusion Detection Systems (NIDS) Overview\n\n"
                "A NIDS monitors network traffic and alerts on suspicious behaviour. "
                "There are two main detection paradigms:\n"
                "1. Signature-based detection: matches traffic against known attack patterns. "
                "Fast and accurate for known attacks; blind to novel threats.\n"
                "2. Anomaly-based detection: establishes a baseline of normal behaviour and "
                "flags deviations. Catches zero-day attacks but has higher false positive rates.\n\n"
                "Hybrid NIDS combines both approaches. This system uses:\n"
                "- Autoencoder (anomaly-based): trained only on normal traffic, flags high "
                "reconstruction error as anomalous.\n"
                "- Isolation Forest (anomaly-based): isolates outlier samples in feature space.\n"
                "- XGBoost arbitrator (supervised / signature-like): trained on labelled data "
                "to resolve uncertain cases.\n\n"
                "Key metrics:\n"
                "- True Positive Rate (Recall): fraction of real attacks detected.\n"
                "- False Positive Rate: fraction of normal traffic incorrectly flagged.\n"
                "- Precision: fraction of flagged alerts that are actual attacks.\n"
                "- F1-score: harmonic mean of precision and recall."
            ),
            "metadata": {"type": "security_concept", "topic": "ids_overview"},
        },
        {
            "id": "concept_dos_guide",
            "text": (
                "Denial of Service (DoS) Attack Detection Guide\n\n"
                "DoS attacks aim to make services unavailable. Categories in NSL-KDD:\n\n"
                "SYN Flood (neptune): Exploits the TCP 3-way handshake. Attacker sends many "
                "SYN packets without completing the handshake, exhausting the server's "
                "connection table. Detected by: very high serror_rate and dst_host_serror_rate, "
                "flag=S0, near-zero dst_bytes.\n\n"
                "Amplification (smurf): ICMP broadcast flood with spoofed source IP. "
                "Detected by: protocol_type=icmp, high count, dst_bytes=0.\n\n"
                "Application flood (back, apache2): Targets specific application "
                "vulnerabilities. Detected by: service=http, high src_bytes, high count.\n\n"
                "Fragmentation (teardrop, pod): Malformed IP fragments. Detected by: "
                "wrong_fragment > 0, low duration.\n\n"
                "General DoS indicators: count > 200, serror_rate > 0.5, "
                "dst_host_serror_rate > 0.5, low duration per connection."
            ),
            "metadata": {"type": "security_concept", "topic": "dos_detection"},
        },
        {
            "id": "concept_probe_guide",
            "text": (
                "Reconnaissance / Probe Attack Detection Guide\n\n"
                "Probe attacks are precursors to actual exploitation — adversaries map the "
                "network landscape before attacking.\n\n"
                "Port scanning (nmap, portsweep): Systematically probes ports on one or more "
                "hosts. Detected by: high diff_srv_rate, flag=S0 or REJ, low dst_bytes, "
                "diverse service distribution.\n\n"
                "Host discovery (ipsweep): Sweeps a range of IPs with ping. Detected by: "
                "high dst_host_count, protocol_type=icmp, very low duration.\n\n"
                "Vulnerability scanning (satan, saint, mscan): Automated tools that test "
                "for known vulnerabilities. Detected by: high srv_count, varied services, "
                "low dst_bytes per connection.\n\n"
                "Mitigation: deploy honeypots to detect scanner IPs, rate-limit connection "
                "attempts, alert on high diff_srv_rate (> 0.3) from a single source."
            ),
            "metadata": {"type": "security_concept", "topic": "probe_detection"},
        },
        {
            "id": "concept_r2l_u2r_guide",
            "text": (
                "R2L and U2R Attack Detection Guide\n\n"
                "R2L (Remote to Local): Attacker gains unauthorized local access from remote.\n"
                "U2R (User to Root): Attacker escalates from user to root privileges.\n\n"
                "These attacks are hardest to detect because they involve fewer connections "
                "and blend with legitimate traffic patterns.\n\n"
                "Credential attacks (guess_passwd, snmpguess): Detected by high "
                "num_failed_logins, service=telnet/ftp/snmp.\n\n"
                "Privilege escalation (buffer_overflow, rootkit): Detected by root_shell=1, "
                "su_attempted=1, num_root > 0. These are critical severity.\n\n"
                "Protocol abuse (httptunnel): Traffic tunnelled over HTTP to bypass firewalls. "
                "Detected by high hot indicator, unusual HTTP payload sizes.\n\n"
                "Response: Implement MFA, principle of least privilege, file integrity "
                "monitoring, monitor for root shell spawns."
            ),
            "metadata": {"type": "security_concept", "topic": "r2l_u2r_detection"},
        },
        {
            "id": "concept_nsl_kdd",
            "text": (
                "NSL-KDD Dataset Overview\n\n"
                "NSL-KDD is the standard benchmark dataset for NIDS research, derived from "
                "the KDD Cup 1999 dataset. It improves on the original by removing duplicate "
                "records and having proportional attack/normal distribution.\n\n"
                "Features (41 + label + difficulty = 43 columns):\n"
                "- Basic features (1-9): connection duration, protocol, service, flag, bytes\n"
                "- Content features (10-22): login attempts, file creations, shell access\n"
                "- Time-based traffic features (23-31): connection rates in 2-second window\n"
                "- Host-based traffic features (32-41): connection rates to same destination\n\n"
                "Attack categories:\n"
                "- DoS (Denial of Service): ~36% of attacks\n"
                "- Probe (Reconnaissance): ~11% of attacks\n"
                "- R2L (Remote to Local): ~1% of attacks (rare, hard to detect)\n"
                "- U2R (User to Root): <0.1% of attacks (very rare, critical)\n\n"
                "Protocol types: tcp, udp, icmp\n"
                "Services: 70 types including http, ftp, smtp, ssh, telnet, dns, etc.\n"
                "Connection flags: SF (normal), S0 (SYN no response), REJ (rejected), "
                "RSTO (reset by originator), RSTOS0, RSTR, S1, S2, S3, OTH"
            ),
            "metadata": {"type": "security_concept", "topic": "nsl_kdd"},
        },
        {
            "id": "concept_response_playbook",
            "text": (
                "Incident Response Playbook for NIDS Alerts\n\n"
                "1. TRIAGE:\n"
                "   - Check confidence level (HIGH/MEDIUM/LOW) and tier used\n"
                "   - HIGH confidence attacks (Tier 1) → immediate action\n"
                "   - LOW confidence (Tier 2, borderline) → investigate first\n\n"
                "2. CONTAINMENT:\n"
                "   - DoS/Probe: Block source IP at perimeter firewall\n"
                "   - R2L: Disable compromised accounts, close exploited service\n"
                "   - U2R: Isolate host immediately, root access compromise is critical\n\n"
                "3. INVESTIGATION:\n"
                "   - Review the triggering features (ae_score, if_score, hybrid_score)\n"
                "   - High ae_score: traffic pattern unusual vs normal baseline\n"
                "   - High if_score: sample is an outlier in feature space\n"
                "   - Both high: high confidence attack\n\n"
                "4. REMEDIATION:\n"
                "   - Apply patches for exploited vulnerabilities\n"
                "   - Update firewall rules\n"
                "   - Reset compromised credentials\n"
                "   - For rootkit/U2R: rebuild system from scratch\n\n"
                "5. LESSONS LEARNED:\n"
                "   - Document false positive/negative patterns\n"
                "   - Adjust detection thresholds if needed\n"
                "   - Update security controls"
            ),
            "metadata": {"type": "security_concept", "topic": "incident_response"},
        },
    ]


# ------------------------------------------------------------------
# 4. Model Explainability Documents
# ------------------------------------------------------------------
def get_explainability_documents() -> List[dict]:
    return [
        {
            "id": "explain_autoencoder",
            "text": (
                "How the Autoencoder Detects Anomalies\n\n"
                "Architecture: Input(116) → Encoder[32→16→8] → Decoder[8→16→32] → Output(116)\n"
                "Training: Trained ONLY on normal traffic samples. The model learns to "
                "compress and reconstruct normal network connection patterns.\n\n"
                "Detection mechanism: When the AE sees normal traffic, it reconstructs it "
                "accurately (low MSE). When it sees an attack, the reconstruction is poor "
                "(high MSE) because the model never learned attack patterns.\n\n"
                "AE Score interpretation:\n"
                "- Score near 0.0: Very similar to normal traffic, low reconstruction error\n"
                "- Score near 0.5: Moderately unusual, may need further investigation\n"
                "- Score near 1.0: Highly anomalous, very different from normal patterns\n\n"
                "Strengths: Excellent at detecting unknown/novel attacks, good precision.\n"
                "Weaknesses: Lower recall — may miss attacks that mimic normal traffic. "
                "Standalone AE achieves ~9% attack recall on NSL-KDD.\n\n"
                "Bottleneck (latent_dim=8): Forces the model to learn the 8 most important "
                "dimensions of normal traffic variation. Anomalies cannot be well-represented "
                "in this compressed space."
            ),
            "metadata": {"type": "model_explain", "model": "autoencoder"},
        },
        {
            "id": "explain_isolation_forest",
            "text": (
                "How the Isolation Forest Detects Anomalies\n\n"
                "Architecture: 200 random decision trees, trained on ALL traffic (normal + attack).\n"
                "contamination=0.1 means the model expects ~10% of training data to be anomalous.\n\n"
                "Detection mechanism: Anomalies are isolated using fewer splits in decision trees "
                "because they occupy sparse regions of the feature space. Anomaly score is based "
                "on the average depth at which a sample is isolated.\n\n"
                "IF Score interpretation:\n"
                "- Score near 0.0: Not isolated easily → likely normal traffic\n"
                "- Score near 0.5: Moderate anomaly score\n"
                "- Score near 1.0: Isolated quickly → strong outlier, likely attack\n\n"
                "Strengths: Effective for detecting outliers in high-dimensional spaces, "
                "fast inference, interpretable concept. Handles DoS and Probe attacks well "
                "due to their distinctive statistical signatures.\n"
                "Weaknesses: Less effective for R2L/U2R attacks that blend with normal traffic "
                "statistically. Contamination parameter must match actual attack rate."
            ),
            "metadata": {"type": "model_explain", "model": "isolation_forest"},
        },
        {
            "id": "explain_hybrid_score",
            "text": (
                "Hybrid Score Computation\n\n"
                "Formula: hybrid_score = 0.6 × ae_score + 0.4 × if_score\n\n"
                "Rationale for weights:\n"
                "- AE (60% weight): Primary detector. Better precision, captures subtle "
                "behavioural anomalies that deviate from normal traffic patterns.\n"
                "- IF (40% weight): Secondary detector. Complements AE by catching statistical "
                "outliers that the AE might reconstruct reasonably well.\n\n"
                "Threshold calibration: The Youden's J statistic is used to find the optimal "
                "decision threshold. It maximises (TPR - FPR), balancing sensitivity with "
                "specificity. threshold = argmax(TPR - FPR) over the ROC curve.\n\n"
                "Performance: The hybrid system achieves ~85% accuracy and 88% attack recall "
                "on NSL-KDD, vs ~57% accuracy for the standalone autoencoder.\n\n"
                "Score range: Both AE and IF scores are normalised to [0, 1] using "
                "MinMaxScaler fitted on the training set."
            ),
            "metadata": {"type": "model_explain", "model": "hybrid"},
        },
        {
            "id": "explain_agreement_layer",
            "text": (
                "Agreement Layer — Tiered Arbitration\n\n"
                "The agreement layer adds a second line of verification to reduce false "
                "positives and flag uncertain detections.\n\n"
                "Tier 1 — Confidence Thresholding:\n"
                "  hybrid_score > 0.75 → ATTACK (HIGH confidence)\n"
                "  hybrid_score < 0.25 → NORMAL (HIGH confidence)\n"
                "  0.25 ≤ score ≤ 0.75 → UNCERTAIN → escalate to Tier 2\n\n"
                "Tier 2 — 3-Model Majority Vote:\n"
                "  Three predictors vote:\n"
                "  1. AE prediction (ae_score ≥ 0.5 → attack)\n"
                "  2. IF prediction (if_score ≥ 0.5 → attack)\n"
                "  3. XGBoost arbitrator (supervised, trained on labelled data)\n"
                "  Final decision = majority (≥ 2/3 agree)\n"
                "  agreement_score = max(n_agree, 3-n_agree) / 3\n\n"
                "Confidence levels in Tier 2:\n"
                "  agreement_score = 1.0 (3/3) → HIGH\n"
                "  agreement_score ≈ 0.67 (2/3) → MEDIUM\n\n"
                "XGBoost arbitrator: A supervised XGBoostClassifier trained on all 25,192 "
                "labelled training samples. Acts as a tiebreaker for borderline cases where "
                "the unsupervised models disagree. Generally achieves >95% accuracy "
                "on NSL-KDD as a standalone classifier."
            ),
            "metadata": {"type": "model_explain", "model": "agreement_layer"},
        },
    ]


# ------------------------------------------------------------------
# Master: return all documents
# ------------------------------------------------------------------
def get_all_documents() -> List[dict]:
    """Return all knowledge base documents combined."""
    docs = []
    docs.extend(get_attack_documents())
    docs.extend(get_mitre_tactic_documents())
    docs.extend(get_security_concept_documents())
    docs.extend(get_explainability_documents())
    return docs
