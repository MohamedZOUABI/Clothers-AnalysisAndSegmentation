# Dockerfile for DeepFashion2 Mask R-CNN Training

# Utilisez une image de base plus récente, mais forcez le tag -cpu pour le test
# REMARQUE : Pour une exécution GPU, utilisez : gcr.io/deeplearning-platform-release/pytorch-x.y:latest-cu11x
FROM gcr.io/deeplearning-platform-release/pytorch-cpu.2-7:latest

# Définir le répertoire de travail
WORKDIR /app

# Copier le code et les dépendances
COPY requirements.txt .
# Assurez-vous que google-cloud-storage et le client GCS sont bien installés
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copier le reste du projet (inclut Step4-train_deepfashion2_maskrcnn.py et Step2_deepfashion2_dataset.py)
COPY . .

# Créer un dossier pour les checkpoints
RUN mkdir -p /app/checkpoints

# Définir le point d'entrée sur le script principal
ENTRYPOINT ["python", "Step4-train_deepfashion2_maskrcnn.py"]