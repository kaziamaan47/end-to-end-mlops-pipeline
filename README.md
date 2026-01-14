#  End-to-End MLOps Pipeline

A complete **end-to-end MLOps project** demonstrating how to build, track, version, deploy, and automate a machine learning model using a **100% free, industry-standard stack**.

This project reflects **real-world MLOps practices**, not just notebook-based experimentation.

---

##  Project Overview

This repository covers the **entire machine learning lifecycle**:

- Data versioning and reproducibility
- Experiment tracking
- Parameterized training pipelines
- Model artifact management
- API-based model serving
- Dockerized deployment
- CI/CD automation with GitHub Actions

---

##  Tech Stack

- **Python**
- **Scikit-learn**
- **DVC** – Data & pipeline versioning
- **MLflow** – Experiment tracking
- **FastAPI** – Model inference API
- **Docker** – Containerization
- **GitHub Actions** – CI/CD
- **Ruff & Pytest** – Linting and testing

---

##  Project Structure

```
end-to-end-mlops-pipeline/
│
├── src/
│   ├── training/          # Model training pipeline
│   └── inference/         # FastAPI inference service
│
├── data/                  # Dataset (DVC-tracked locally)
├── models/                # Trained model artifact
├── tests/                 # CI tests
│
├── dvc.yaml               # Reproducible pipeline definition
├── params.yaml            # Training parameters
├── Dockerfile             # API container definition
├── requirements.api.txt   # Runtime dependencies
├── requirements.ci.txt    # CI dependencies
└── README.md
```

---

##  Reproducible Training

Run the full training pipeline with a single command:

```bash
dvc repro
```

This pipeline:

- Loads the dataset
- Trains the machine learning model
- Logs parameters and metrics to **MLflow**
- Saves versioned model artifacts

---

##  Experiment Tracking (MLflow)

Launch the MLflow UI locally:

```bash
mlflow ui
```

Open in your browser:

```
http://127.0.0.1:5000
```

You can inspect:
- Experiment runs
- Hyperparameters
- Accuracy and evaluation metrics
- Stored model artifacts

---

##  Model Serving (FastAPI)

Run the inference API locally using Docker.

### Build the Docker image
```bash
docker build -t iris-api .
```

### Run the container
```bash
docker run -p 8000:8000 iris-api
```

### Access Swagger UI
```
http://127.0.0.1:8000/docs
```

Use the `/predict` endpoint to send feature values and receive predictions.

---

##  CI/CD with GitHub Actions

The project includes automated workflows that:

- Run linting with **Ruff**
- Execute tests with **Pytest**
- Validate Docker image builds

All workflows run automatically on every push and pull request.

---

##  Key Learnings

- Designing reproducible ML pipelines using DVC
- Separating development, CI, and runtime dependencies
- Handling real-world CI/CD failures and fixes
- Dockerizing ML inference services
- Applying MLOps best practices beyond notebooks

---

## 👤 Author

**Amaan Kazi**  
Master’s Student | Machine Learning & MLOps  
GitHub: https://github.com/kaziamaan47
