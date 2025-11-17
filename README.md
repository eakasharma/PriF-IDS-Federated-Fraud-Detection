PriF-IDS: A Privacy-Preserving Fraud Detection System via Federated Learning

This project is a proof-of-concept (PoC). It implements a federated learning (FL) system to train a fraud detection model without ever exposing or centralizing sensitive client data.

This repository contains the complete, working code and documentation for the project.

► The Problem

Financial institutions face a critical challenge:

They need to use AI to detect sophisticated, real-time fraud.

They must comply with strict data privacy laws (like GDPR) that forbid centralizing sensitive customer data.

The traditional AI model of "pool all data in one place" is no longer a safe or legal option.

► The Solution: Federated Learning

This project demonstrates a solution using Federated Learning (FL).

The system is composed of a central Server and multiple Clients (simulating 3 different banks).

No Data Sharing: The clients never send their private data to the server.

Local Training: Each client trains a model on its own local, private data.

Secure Aggregation: Clients only send their anonymous model learnings (the "weights" or "parameters") to the server.

Global Model: The server averages these learnings to build a single, robust "global model" that has learned from all clients without seeing any of their data.

► Technology Stack

Core Language: Python 3.11

Federation Framework: flwr (Flower)

ML/Data Libraries: scikit-learn, pandas, numpy

Dataset: Credit Card Fraud Detection (Kaggle)(https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

► How to Run This Project

This project runs in a multi-process "cross-silo" simulation. You will need 4 separate terminal windows.

1. Setup the Environment

# 1. Clone this repository (or download the ZIP)
git clone [https://github.com/eakasharma/PriF-IDS-Federated-Fraud-Detection.git](https://github.com/eakasharma/PriF-IDS-Federated-Fraud-Detection.git)
cd PriF-IDS-Federated-Fraud-Detection

# 2. Create a Python virtual environment
python -m venv venv  
source venv/bin/activate  # On Mac/Linux  
.\venv\Scripts\activate  # On Windows

# 3. Install the required libraries
pip install -r requirements.txt


2. Download the Data

This project requires the creditcard.csv dataset.

Download it from Kaggle. (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Place the creditcard.csv file in the main PriF-IDS-Federated-Fraud-Detection folder.

3. Run the Simulation

(You must have your venv activated in all 4 terminals)

Terminal 1 (The Server):
Start the server and wait for clients to connect.

python server.py


Terminal 2 (Client 0):

python client.py --partition 0


Terminal 3 (Client 1):

python client.py --partition 1


Terminal 4 (Client 2):

python client.py --partition 2


As soon as all 3 clients connect, the training will begin. You can watch the "Server-Side Evaluation" in Terminal 1 as the model learns.

► Understanding the Results

This project is a successful demonstration of Iterative Refinement. The warm_start=True parameter ensures the model learns from the global parameters in each round, not just from its own data.

Final Run Output (Example):

--- Server-Side Evaluation (Round 0) ---
Loss: 0.6931 | Precision: 0.0000 | Recall: 0.0000 | F1: 0.0000

--- Server-Side Evaluation (Round 1) ---
Loss: 0.0972 | Precision: 0.0661 | Recall: 0.9184 | F1: 0.1233

--- Server-Side Evaluation (Round 2) ---
Loss: 0.0966 | Precision: 0.0647 | Recall: 0.9184 | F1: 0.1210

--- Server-Side Evaluation (Round 3) ---
Loss: 0.0986 | Precision: 0.0626 | Recall: 0.9184 | F1: 0.1172

--- Server-Side Evaluation (Round 4) ---
Loss: 0.0965 | Precision: 0.0641 | Recall: 0.9184 | F1: 0.1198

--- Server-Side Evaluation (Round 5) ---
Loss: 0.0968 | Precision: 0.0645 | Recall: 0.9184 | F1: 0.1205


Analysis: The Precision vs. Recall Trade-off

The most important part of this project is understanding the metrics.

Recall: The model is successfully catching 91.8% of all fraud! This is excellent.

Precision: To achieve this, it's very "cautious," flagging many non-fraudulent transactions. Only 6.4% of its alerts are correct.

This is a classic High-Recall Model. For fraud detection, this is a good outcome. The bank would rather review 100 false alerts (low precision) than miss one multi-million dollar fraudulent transaction (low recall).

This result was achieved intentionally by using class_weight="balanced" in the code, which tells the model to prioritize finding the rare "fraud" class.
