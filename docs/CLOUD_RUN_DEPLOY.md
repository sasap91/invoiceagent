# Deploy ProcureAgent to Google Cloud Run

This is the preferred permanent host when the team does not want Streamlit Community Cloud. Firebase projects are Google Cloud projects underneath, but **Firebase Hosting alone cannot run this Python/Tesseract/PyTorch app**. Cloud Run builds the committed `Dockerfile`, runs the container, and returns a stable HTTPS URL.

## One-time project decision

Use a dedicated project; do not reuse an unrelated production Firebase project.

```bash
gcloud projects create procureagent-sundai-2026 \
  --name="ProcureAgent Sundai 2026"
gcloud billing projects link procureagent-sundai-2026 \
  --billing-account=YOUR_APPROVED_BILLING_ACCOUNT
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  --project=procureagent-sundai-2026
```

Creating and billing-linking the project are external account changes. Run them only after the project owner approves the exact billing account.

## Deploy the pushed repository state

From a clean clone of `sasap91/invoiceagent`:

```bash
git switch main
git pull --ff-only
gcloud run deploy procureagent \
  --project=procureagent-sundai-2026 \
  --region=us-east1 \
  --source=. \
  --allow-unauthenticated \
  --memory=4Gi \
  --cpu=2 \
  --timeout=3600 \
  --min=1 \
  --max=2
```

`--source=.` uses Cloud Build, so local Docker is not required. `--min=1` keeps one instance warm for the event and incurs cost until changed back to zero.

## Warm and verify

1. Open the returned `https://…run.app` URL in a private browser.
2. Choose the bundled Fresh Farms invoice and select **Fresh Farms**.
3. Run the invoice reader once. The first run may populate the pinned Hugging Face cache.
4. Complete all four steps through simulated `PAID_CONFIRMED`.
5. Confirm the page shows `Second cash deduction $0.00`.
6. Test the same URL at phone width.

No secret is required for the bundled demo. Do not add `FAL_KEY`, banking credentials, or private invoices.

## After the event

To stop paying for an always-warm instance while retaining the URL:

```bash
gcloud run services update procureagent \
  --project=procureagent-sundai-2026 \
  --region=us-east1 \
  --min=0
```

Cloud Run supports WebSockets, which Streamlit uses. Keep the app on the direct Cloud Run URL instead of placing it behind Firebase Hosting's request rewrite.

Official references:

- <https://cloud.google.com/run/docs/deploying-source-code>
- <https://cloud.google.com/run/docs/triggering/websockets>
- <https://firebase.google.com/docs/hosting/cloud-run>
