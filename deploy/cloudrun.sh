#!/usr/bin/env bash
# One-shot provisioning + deployment of Vlogsmith to Cloud Run. Safe to re-run.
#
#   PROJECT_ID=my-project ./deploy/cloudrun.sh
#
# Options (env vars):
#   REGION         us-central1
#   SERVICE        vlogsmith
#   BUCKET         <project>-vlogsmith
#   GEMINI_AUTH    apikey | vertex   (default apikey: GOOGLE_API_KEY from ENV_FILE is stored in Secret Manager)
#   ENV_FILE       ./.env            (source of GOOGLE_API_KEY and MODEL_* values)
#   ACCESS         iap | public | private   (default iap: Google sign-in, only IAP_USERS may open it)
#   IAP_USERS      comma-separated emails  (default: the active gcloud account)
#   MIN_INSTANCES  0 (scale to zero)   MAX_INSTANCES 5
set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-vlogsmith}"
BUCKET="${BUCKET:-${PROJECT_ID}-vlogsmith}"
GEMINI_AUTH="${GEMINI_AUTH:-apikey}"
ACCESS="${ACCESS:-iap}"
MIN_INSTANCES="${MIN_INSTANCES:-0}"
MAX_INSTANCES="${MAX_INSTANCES:-5}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${ENV_FILE:-${ROOT}/.env}"
SA_NAME="${SERVICE}-runtime"
SA="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
SECRET="${SERVICE}-gemini-api-key"
IAP_USERS="${IAP_USERS:-$(gcloud config get-value account 2>/dev/null)}"

gc() { gcloud --project "$PROJECT_ID" --quiet "$@"; }
env_value() { [[ -f "$ENV_FILE" ]] && grep -E "^$1=" "$ENV_FILE" | tail -1 | cut -d= -f2- | sed -E 's/[[:space:]]+#.*$//; s/^"//; s/"$//' || true; }

MODEL_FAST="${MODEL_FAST:-$(env_value MODEL_FAST)}"
MODEL_SMART="${MODEL_SMART:-$(env_value MODEL_SMART)}"
MODEL_TTS="${MODEL_TTS:-$(env_value MODEL_TTS)}"

echo "==> Enabling APIs"
APIS=(run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com firestore.googleapis.com
      storage.googleapis.com iamcredentials.googleapis.com secretmanager.googleapis.com)
[[ "$GEMINI_AUTH" == "vertex" ]] && APIS+=(aiplatform.googleapis.com)
[[ "$ACCESS" == "iap" ]] && APIS+=(iap.googleapis.com cloudresourcemanager.googleapis.com)
gc services enable "${APIS[@]}"

echo "==> Service account ${SA}"
gc iam service-accounts describe "$SA" >/dev/null 2>&1 || \
  gc iam service-accounts create "$SA_NAME" --display-name "Vlogsmith runtime"
ROLES=(roles/datastore.user roles/logging.logWriter)
[[ "$GEMINI_AUTH" == "vertex" ]] && ROLES+=(roles/aiplatform.user)
for role in "${ROLES[@]}"; do
  gc projects add-iam-policy-binding "$PROJECT_ID" --member "serviceAccount:${SA}" --role "$role" --condition=None >/dev/null
done
# Lets the service sign GCS upload/download URLs through the IAM signBlob API (no key files).
gc iam service-accounts add-iam-policy-binding "$SA" --member "serviceAccount:${SA}" \
  --role roles/iam.serviceAccountTokenCreator >/dev/null

echo "==> Bucket gs://${BUCKET}"
gcloud storage buckets describe "gs://${BUCKET}" --project "$PROJECT_ID" >/dev/null 2>&1 || \
  gcloud storage buckets create "gs://${BUCKET}" --project "$PROJECT_ID" --location "$REGION" \
    --uniform-bucket-level-access --public-access-prevention
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" --member "serviceAccount:${SA}" \
  --role roles/storage.objectAdmin >/dev/null
gcloud storage buckets update "gs://${BUCKET}" --lifecycle-file "${ROOT}/deploy/lifecycle.json" >/dev/null

echo "==> Firestore"
gc firestore databases describe --database="(default)" >/dev/null 2>&1 || \
  gc firestore databases create --database="(default)" --location "$REGION" --type firestore-native

ENV_VARS="ENVIRONMENT=cloud,STORAGE_BACKEND=gcs,JOBSTORE_BACKEND=firestore,GCS_BUCKET=${BUCKET},GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_LOCATION=${REGION},MAX_CONCURRENT_JOBS=2,RENDER_CONCURRENCY=3,WORK_DIR=/tmp/vlogsmith-work,COMPUTE_VCPU=4,COMPUTE_MEMORY_GIB=8"
[[ -n "$MODEL_FAST" ]] && ENV_VARS+=",MODEL_FAST=${MODEL_FAST}"
[[ -n "$MODEL_SMART" ]] && ENV_VARS+=",MODEL_SMART=${MODEL_SMART}"
[[ -n "$MODEL_TTS" ]] && ENV_VARS+=",MODEL_TTS=${MODEL_TTS}"
SECRET_ARGS=()
if [[ "$GEMINI_AUTH" == "apikey" ]]; then
  echo "==> Secret Manager: ${SECRET}"
  KEY="${GOOGLE_API_KEY:-$(env_value GOOGLE_API_KEY)}"
  [[ -n "$KEY" ]] || { echo "GOOGLE_API_KEY not found in ${ENV_FILE}" >&2; exit 1; }
  gc secrets describe "$SECRET" >/dev/null 2>&1 || gc secrets create "$SECRET" --replication-policy automatic >/dev/null
  CURRENT="$(gc secrets versions access latest --secret "$SECRET" 2>/dev/null || true)"
  if [[ "$CURRENT" != "$KEY" ]]; then
    printf '%s' "$KEY" | gc secrets versions add "$SECRET" --data-file=- >/dev/null
    echo "    stored a new key version"
  fi
  unset KEY CURRENT
  gc secrets add-iam-policy-binding "$SECRET" --member "serviceAccount:${SA}" \
    --role roles/secretmanager.secretAccessor >/dev/null
  ENV_VARS+=",GOOGLE_GENAI_USE_VERTEXAI=FALSE"
  SECRET_ARGS=(--set-secrets "GOOGLE_API_KEY=${SECRET}:latest")
else
  ENV_VARS+=",GOOGLE_GENAI_USE_VERTEXAI=TRUE"
fi

case "$ACCESS" in
  public) ACCESS_ARGS=(--allow-unauthenticated) ;;
  iap) ACCESS_ARGS=(--no-allow-unauthenticated --iap) ;;
  *) ACCESS_ARGS=(--no-allow-unauthenticated) ;;
esac

echo "==> Deploying ${SERVICE} (Cloud Build from source; takes a few minutes)"
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
  "${ACCESS_ARGS[@]}" "${SECRET_ARGS[@]}" \
  --set-env-vars "$ENV_VARS"

URL="$(gc run services describe "$SERVICE" --region "$REGION" --format 'value(status.url)')"
# Cloud Run serves the same service on several run.app hostnames; allow uploads from all of them.
ORIGINS="$(gc run services describe "$SERVICE" --region "$REGION" --format json | python3 -c '
import json, sys
svc = json.load(sys.stdin)
urls = json.loads(svc["metadata"].get("annotations", {}).get("run.googleapis.com/urls", "[]") or "[]")
urls.append(svc["status"]["url"])
print(json.dumps(sorted(set(urls))))')"

if [[ "$ACCESS" == "iap" ]]; then
  echo "==> IAP access"
  NUM="$(gc projects describe "$PROJECT_ID" --format 'value(projectNumber)')"
  gc beta services identity create --service=iap.googleapis.com >/dev/null 2>&1 || true
  gc run services add-iam-policy-binding "$SERVICE" --region "$REGION" \
    --member "serviceAccount:service-${NUM}@gcp-sa-iap.iam.gserviceaccount.com" --role roles/run.invoker >/dev/null
  IFS=',' read -ra USERS <<< "$IAP_USERS"
  for u in "${USERS[@]}"; do
    [[ -z "$u" ]] && continue
    gc iap web add-iam-policy-binding --resource-type=cloud-run --service="$SERVICE" --region="$REGION" \
      --member "user:${u}" --role roles/iap.httpsResourceAccessor >/dev/null
    echo "    granted ${u}"
  done
fi

echo "==> Bucket CORS for browser uploads from ${URL}"
CORS_FILE="$(mktemp)"
cat > "$CORS_FILE" <<JSON
[{"origin": ${ORIGINS}, "method": ["PUT", "GET", "HEAD"],
  "responseHeader": ["Content-Type", "Content-Length", "Range"], "maxAgeSeconds": 3600}]
JSON
gcloud storage buckets update "gs://${BUCKET}" --cors-file "$CORS_FILE" >/dev/null
rm -f "$CORS_FILE"

echo
echo "Deployed: ${URL}"
case "$ACCESS" in
  iap) echo "Sign in with: ${IAP_USERS}. Add people with:"
       echo "  gcloud iap web add-iam-policy-binding --resource-type=cloud-run --service=${SERVICE} --region=${REGION} --member=user:EMAIL --role=roles/iap.httpsResourceAccessor --project ${PROJECT_ID}" ;;
  private) echo "Private: open it through 'gcloud run services proxy ${SERVICE} --region ${REGION}'" ;;
esac
