"""End-to-end demo: small data -> train -> score two txns (normal vs scam)."""
from data_gen.generate import generate
from features.build import build_features
from models.graph import build_graph, fraud_nodes, network_risk
from models.infer import load_artifacts, score_features
from models.train import train


def main():
    customers, devices, txns = generate(n_customers=400, n_txns=4000, seed=7)
    txns.to_csv("/tmp/demo_tx.csv", index=False)
    import pandas as pd
    customers.to_csv("/tmp/demo_c.csv", index=False)
    devices.to_csv("/tmp/demo_d.csv", index=False)
    print("demo data:", len(txns), "fraud:", int(txns.is_fraud.sum()))
