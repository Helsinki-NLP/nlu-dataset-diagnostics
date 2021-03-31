# BUCKET_NAME: unique bucket name
BUCKET_NAME=nlu-dataset-diagnostics-001

# The PyTorch image provided by AI Platform Training.
IMAGE_URI=gcr.io/cloud-ml-public/training/pytorch-gpu.1-4

# JOB_NAME: the name of your job running on AI Platform.
JOB_NAME=nli_diagnostics_job_mnli_both_$(date +%Y%m%d_%H%M%S)

echo "Submitting AI Platform Training job: ${JOB_NAME}"

PACKAGE_PATH=./trainer # this can be a GCS location to a zipped and uploaded package

REGION=us-central1

# JOB_DIR: Where to store prepared package and upload output model.
JOB_DIR=gs://${BUCKET_NAME}/${JOB_NAME}/models

gcloud ai-platform jobs submit training ${JOB_NAME} \
    --region ${REGION} \
    --master-image-uri ${IMAGE_URI} \
    --config config.yaml \
    --job-dir ${JOB_DIR} \
    --module-name trainer.task \
    --package-path ${PACKAGE_PATH} \
    -- \
    --epochs 2 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --corrupt_train \
    --corrupt_test \
    --pos 'NOUN'
