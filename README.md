Deutsch-Digest 🇩🇪
Automated MLOps Pipeline for German Language Summarization

Deutsch-Digest is a production-grade NLP tool designed to condense complex German text into concise, coherent summaries. It leverages state-of-the-art multilingual transformer models and follows a full MLOps lifecycle for reliable experiment tracking and pipeline orchestration.

Key Features

Abstractive Summarization: Uses a fine-tuned mT5 (Multilingual T5) transformer model to generate original, human-like summaries rather than just extracting existing sentences.

Cross-Lingual Verification: Includes an integrated verification step using deep-translator to translate summaries into English, ensuring context retention and factual accuracy for non-native users.

MLOps Orchestration: Full pipeline management using Apache Airflow for DAG-based workflows and MLflow for precise experiment tracking and hyperparameter logging.

Streamlit UI: A clean, interactive web interface for real-time inference and visualization.

Tech Stack

Language: Python 3.13.

NLP Frameworks: Hugging Face Transformers, PyTorch, SentencePiece.

MLOps Tools: Apache Airflow (Orchestration), MLflow (Tracking), Git LFS.

Frontend & Deployment: Streamlit Cloud, Docker.

Technical Workflow

Data Ingestion: Automated ingestion and preprocessing of German text via Airflow DAGs.

Experiment Tracking: Hyperparameters and ROUGE-L metrics are logged via MLflow to ensure reproducibility and model quality.

Validation: Uses a back-translation layer to provide a "semantic logic check" between the source and summary.

Infrastructure: Managed repository using Git LFS to handle large model weights and binary files.
