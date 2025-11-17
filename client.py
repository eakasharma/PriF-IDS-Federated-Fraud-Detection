import flwr as fl
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, precision_recall_fscore_support
import warnings
import argparse 

warnings.filterwarnings("ignore", category=UserWarning)

# 1. DATA LOADING (CLIENT-SIDE)
def load_client_data(partition_id: int, num_clients: int):
    data = pd.read_csv("creditcard.csv")
    
    scaler = StandardScaler()
    data["NormalizedAmount"] = scaler.fit_transform(data["Amount"].values.reshape(-1, 1))
    data = data.drop(["Time", "Amount"], axis=1)
    
    X = data.drop("Class", axis=1)
    y = data["Class"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    print(f"Loading data partition {partition_id}...")
    
    X_partitions = np.array_split(X_train, num_clients)
    y_partitions = np.array_split(y_train, num_clients)
    
    X_local = X_partitions[partition_id]
    y_local = y_partitions[partition_id]
    
    return X_local, y_local

# 2. DEFINE THE FLOWER CLIENT (Methodology Step 1)
class FraudClient(fl.client.NumPyClient):
    
    def __init__(self, X_local, y_local):
        self.X = X_local
        self.y = y_local
        
        self.model = LogisticRegression(
            solver="saga", 
            max_iter=100, 
            class_weight="balanced", 
            random_state=42,
            warm_start=True  
        )
        
        # Pre-initialize the model's parameters
        n_features = X_local.shape[1]
        self.model.classes_ = np.array([0, 1]) 
        self.model.coef_ = np.zeros((1, n_features))
        self.model.intercept_ = np.zeros((1,))

    def get_parameters(self, config):
        print(f"[Client {args.partition}] Getting parameters")
        return [self.model.coef_, self.model.intercept_]

    def set_parameters(self, parameters):
        print(f"[Client {args.partition}] Setting parameters")
        self.model.coef_ = parameters[0]
        self.model.intercept_ = parameters[1]

    def fit(self, parameters, config):
        print(f"[Client {args.partition}] Fitting model on {len(self.X)} samples")
        self.set_parameters(parameters)
        
        # Because warm_start=True, this continues training
        self.model.fit(self.X, self.y)
        
        return self.get_parameters(config), len(self.X), {}

    def evaluate(self, parameters, config):
        print(f"[Client {args.partition}] Evaluating model")
        self.set_parameters(parameters)
        
        loss = log_loss(self.y, self.model.predict_proba(self.X))
        y_pred = self.model.predict(self.X)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y, y_pred, average='binary', zero_division=0
        )
        return loss, len(self.X), {"local_f1": f1}


# 3. START THE CLIENT
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--partition",
        type=int,
        required=True,
        choices=[0, 1, 2],
        help="The partition ID for this client (0, 1, or 2)."
    )
    args = parser.parse_args()
    
    X_local, y_local = load_client_data(partition_id=args.partition, num_clients=3)
    client = FraudClient(X_local, y_local)
    
    print(f"Starting Client {args.partition}...")
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080", 
        client=client,
    )
    print(f"Client {args.partition} finished.")