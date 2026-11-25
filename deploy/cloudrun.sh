#!/usr/bin/env bash
# One-shot provisioning + deployment of Vlogsmith to Cloud Run.
#
#   PROJECT_ID=my-project ./deploy/cloudrun.sh
#
# Optional env: REGION (us-central1), SERVICE (vlogsmith), BUCKET (<project>-vlogsmith),
#               PUBLIC (false), MAX_INSTANCES (5), MIN_INSTANCES (1)
# Re-running is safe: every step is idempotent.
set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-vlogsmith}"
BUCKET="${BUCKET:-${PROJECT_ID}-vlogsmith}"
PUBLIC="${PUBLIC:-false}"
MAX_INSTANCES="${MAX_INSTANCES:-5}"
MIN_INSTANCES="${MIN_INSTANCES:-1}"
SA_NAME="${SERVICE}-runtime"
SA="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

gc() { gcloud --project "$PROJECT_ID" --quiet "$@"; }

echo "==> Enabling APIs"
gc services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com \
  aiplatform.googleapis.com firestore.googleapis.com storage.googleapis.com iamcredentials.googleapis.com

echo "==> Service account ${SA}"
gc iam service-accounts describe "$SA" >/dev/null 2>&1 || \
  gc iam service-accounts create "$SA_NAME" --display-name "Vlogsmith runtime"
for role in roles/aiplatform.user roles/datastore.user roles/logging.logWriter; do
  gc projects add-iam-policy-binding "$PROJECT_ID" --member "serviceAccount:${SA}" --role "$role" --condition=None >/dev/null
done
# Needed to sign GCS upload/download URLs via the IAM signBlob API (no key files).
gc iam service-accounts add-iam-policy-binding "$SA" --member "serviceAccount:${SA}" \
  --role roles/iam.serviceAccountTokenCreator >/dev/null

echo "==> Bucket gs://${BUCKET}"
gcloud storage buckets describe "gs://${BUCKET}" --project "$PROJECT_ID" >/dev/null 2>&1 || \
  gcloud storage buckets create "gs://${BUCKET}" --project "$PROJECT_ID" --location "$REGION" \
    --uniform-bucket-level-access --public-access-prevention
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" --member "serviceAccount:${SA}" \
  --role roles/storage.objectAdmin >/dev/null
gcloud storage buckets update "gs://${BUCKET}" --lifecycle-file "${ROOT}/deploy/lifecycle.json"

echo "==> Firestore"
gc firestore databases describe --database="(default)" >/dev/null 2>&1 || \
  gc firestore databases create --database="(default)" --location "$REGION" --type firestore-native

echo "==> Deploying ${SERVICE} (Cloud Build from source)"
AUTH_FLAG="--no-allow-unauthenticated"
[[ "$PUBLIC" == "true" ]] && AUTH_FLAG="--allow-unauthenticated"
gc run deploy "$SERVICE" \
  --source "$ROOT" \
  --region "$REGION" \
  --service-account "$SA" \
  --execution-environment gen2 \
  --cpu 4 --memory 8Gi \
  --no-cpu-throttling \
  --timeout 3600 \
  --concurrency 40 \
  --min-instances "$MIN_INSTANCES" --max-instances "$MAX_INSTANCES" \
  "$AUTH_FLAG" \
  --set-env-vars "ENVIRONMENT=cloud,STORAGE_BACKEND=gcs,JOBSTORE_BACKEND=firestore,GCS_BUCKET=${BUCKET},GOOGLE_GENAI_USE_VERTEXAI=TRUE,GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_LOCATION=${REGION},MAX_CONCURRENT_JOBS=2,RENDER_CONCURRENCY=3,WORK_DIR=/tmp/vlogsmith-work"

URL="$(gc run services describe "$SERVICE" --region "$REGION" --format 'value(status.url)')"

echo "==> Allowing browser uploads from ${URL} (bucket CORS)"
CORS_FILE="$(mktemp)"
cat > "$CORS_FILE" <<JSON
[{"origin": ["${URL}"], "method": ["PUT", "GET", "HEAD"],
  "responseHeader": ["Content-Type", "Content-Length", "Range"], "maxAgeSeconds": 3600}]
JSON
gcloud storage buckets update "gs://${BUCKET}" --cors-file "$CORS_FILE"
rm -f "$CORS_FILE"

echo
echo "Deployed: ${URL}"
if [[ "$PUBLIC" != "true" ]]; then
  echo "The service is private. Grant users access with:"
  echo "  gcloud run services add-iam-policy-binding ${SERVICE} --region ${REGION} --member user:someone@example.com --role roles/run.invoker"
  echo "or put it behind IAP / a load balancer, or redeploy with PUBLIC=true."
fi
