import flwr as fl
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, precision_recall_fscore_support
import warnings
from typing import Dict, Tuple

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 1. DATA LOADING (SERVER-SIDE)
def load_test_data():
    data = pd.read_csv("creditcard.csv")
    
    scaler = StandardScaler()
    data["NormalizedAmount"] = scaler.fit_transform(data["Amount"].values.reshape(-1, 1))
    data = data.drop(["Time", "Amount"], axis=1)
    
    X = data.drop("Class", axis=1)
    y = data["Class"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    
    print("Global test set loaded for server-side evaluation.")
    return X_test, y_test

# 2. SERVER-SIDE EVALUATION (Methodology Step 3)
def get_server_evaluate_fn(X_test, y_test):
    n_features = X_test.shape[1] 

    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]):
        
        model = LogisticRegression(
            solver="saga", 
            max_iter=100, 
            class_weight="balanced", 
            random_state=42,
            warm_start=True 
        )
        
        # Manually set the parameters' shape
        model.classes_ = np.array([0, 1]) 
        model.coef_ = np.zeros((1, n_features))
        model.intercept_ = np.zeros((1,))
        
        # Set aggregated parameters
        model.coef_ = parameters[0]
        model.intercept_ = parameters[1]
        
        # Evaluate
        loss = log_loss(y_test, model.predict_proba(X_test))
        y_pred = model.predict(X_test)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary', zero_division=0
        )
        
        print("\n--- Server-Side Evaluation (Round {server_round}) ---")
        print(f"Loss: {loss:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        
        return loss, {"global_precision": precision, "global_recall": recall, "global_f1": f1}

    return evaluate

# 3. START THE SERVER
if __name__ == "__main__":
    
    NUM_ROUNDS = 5
    X_test, y_test = load_test_data()
    
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  
        min_fit_clients=3, 
        min_available_clients=3, 
        evaluate_fn=get_server_evaluate_fn(X_test, y_test), 
    )

    print("Starting Federated Learning Server...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )
    print("Server finished.")