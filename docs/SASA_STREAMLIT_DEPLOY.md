# Sasa — deploy InvoiceAgent to a permanent Streamlit URL

> **Optional fallback:** the team currently prefers the containerized [Google Cloud Run path](CLOUD_RUN_DEPLOY.md). Use this checklist only if the team decides to use Streamlit Community Cloud after all.

Wilson has prepared the repository for Streamlit Community Cloud. Because you administer `sasap91/invoiceagent`, please perform the one-time app creation below. After that, pushes to `main` redeploy automatically.

## One-time deployment

1. Open <https://share.streamlit.io> and sign in with the GitHub account that administers `sasap91/invoiceagent`.
2. If Streamlit asks for GitHub access, authorize the `sasap91/invoiceagent` repository.
3. Select **Create app**.
4. Use these settings:

   | Setting | Value |
   |---|---|
   | Repository | `sasap91/invoiceagent` |
   | Branch | `main` |
   | Main file path | `procure_app.py` |
   | Python version | `3.12` |

5. Choose a memorable app URL if Streamlit offers one, such as `invoiceagent` or `invoice-agent-demo`.
6. Do **not** add `FAL_KEY`, banking credentials, invoice data, or any other secret. The recording demo uses committed synthetic fixtures and does not need a secret.
7. Select **Deploy** and keep the build log open. `packages.txt` installs Tesseract; `requirements.txt` installs Streamlit, Pillow, Torch, Transformers, PEFT, and the repository package.
8. Send the resulting `https://...streamlit.app` URL to Wilson and the team.

## Warm it before the presentation

The first model run downloads the pinned LayoutLMv3 base and Ryan's adapter. Do this while reliable internet is available:

1. Open the permanent URL.
2. Choose **Guided demo**.
3. Select the bundled Fresh Farms invoice.
4. Run the invoice-reading step once.
5. Confirm that the proposed invoice number is `FF-10482` and that the safety gate requests human review when confidence is below the frozen threshold.
6. Complete the simulated approval and bundled receipt path once, ending at simulated `PAID_CONFIRMED`.
7. Refresh the app and confirm it loads again.

## Five-minute acceptance check

- The app opens in a private/incognito browser window.
- The bundled invoice and receipt images render.
- Invoice OCR reports real Tesseract output.
- The pinned LayoutLMv3 specialist adapted by Ryan proposes `FF-10482`.
- The screen clearly says the amount and restaurant context come from OCR/rules or synthetic lookup—not LayoutLMv3.
- No state changes before explicit human/operator approval.
- The receipt matches supplier, invoice, full amount, and currency.
- The final state says simulated `PAID_CONFIRMED` and **no real money moved**.
- The technical/code panels open for engineers.
- The app works at both desktop and phone widths.

## If deployment fails

Copy the Streamlit build/runtime log and send it to Wilson without including secrets.

Do not weaken the model gate, remove the local model, or relabel fixture evidence as live just to make deployment pass. Streamlit Community Cloud has limited memory and can hibernate idle apps. If the Torch/LayoutLMv3 runtime exceeds the available memory, keep the current Cloudflare Quick Tunnel for the event and move the same Docker image to a deliberately selected Google Cloud Run project with at least 4 GiB RAM after the team approves the billed project.

Official deployment guide: <https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy>
