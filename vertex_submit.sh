#!/bin/bash

# -----------------------------
# Configuration du projet GCP
# -----------------------------
PROJECT_ID="ai-training-env" # <<< CHANGEZ CECI
REGION="europe-west1"
REPO_NAME="ml-images" # Nom du dépôt Artifact Registry (par défaut)
IMAGE_TAG="maskrcnn-cpu-test"

# L'URI de l'image Docker dans Artifact Registry
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_TAG}:latest"

# MODIFIÉ: Nom de job plus spécifique pour le test rapide
JOB_NAME="maskrcnn-cpu-quick-test-$(date +%Y%m%d-%H%M%S)"

# Buckets GCS
DATASET_BUCKET="gs://ai-models-dataset/deepfashion2" # Assurez-vous que le dossier 'deepfashion2' est bien à la racine du bucket ou ajustez le chemin
OUTPUT_BUCKET="gs://ai-models-output/maskrcnn-results"

# -----------------------------
# Préparation Artifact Registry (si le dépôt n'existe pas)
# -----------------------------
echo "Vérification/Création du dépôt Artifact Registry : $REPO_NAME"
gcloud artifacts repositories describe $REPO_NAME \
  --location=$REGION \
  --project=$PROJECT_ID \
  --format="value(name)" 2>/dev/null || \
  gcloud artifacts repositories create $REPO_NAME \
    --repository-format=docker \
    --location=$REGION \
    --project=$PROJECT_ID

# -----------------------------
# Build et push du container Docker
# -----------------------------
echo "📦 Construction et envoi du container avec le tag :$IMAGE_TAG"

# Authentification Docker auprès d'Artifact Registry
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Construction et push en une seule commande pour Artifact Registry
docker build -t ${IMAGE_URI} .
docker push ${IMAGE_URI}

# -----------------------------
# Lancer le job Vertex AI en mode CPU (Quick Test)
# -----------------------------
echo "🚀 Soumission du job Vertex AI (CPU Quick Test - n1-standard-4)..."

# MACHINE_TYPE=n1-standard-4 est correct pour un test CPU. 
# L'option --accelerator est omise volontairement.
gcloud ai custom-jobs create \
  --project=$PROJECT_ID \
  --region=$REGION \
  --display-name=$JOB_NAME \
  --worker-pool-spec=machine-type=n1-standard-4,replica-count=1,container-image-uri=$IMAGE_URI \
  --args="--dataset_bucket=$DATASET_BUCKET","--output_bucket=$OUTPUT_BUCKET"

echo "✅ Job soumis avec succès : $JOB_NAME"
echo "➡️ Consultez les logs et le statut ici : https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=$PROJECT_ID"