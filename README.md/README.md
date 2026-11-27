# OCT Retinal Disease Detection - Streamlit dashboard (Steps 1-9)

Educational prototype. Not a clinical diagnostic tool.

## Run
    pip install -r requirements.txt
    streamlit run app.py

Put these files next to `app.py` (or upload them in the sidebar under "Model files"):
- `step6_best_model.pt` (+ optional `optimization_config.json`) - tuned ResNet18 from Step 6
- `step4_custom_cnn.pt` (+ optional `custom_cnn_config.json`) - Step 4 custom CNN (built-in defaults: 64 px, width 16, fc 128, tanh, avg-pool)

## Pages
Home, **Try the model (Step 9)** (multi-model upload, confidence, Grad-CAM, CSV export), Steps 1-8 (your recorded results),
**Step 7 > Model efficiency** (size / speed / accuracy table, chart, live CPU benchmark of loaded models), About and limitations.

## Notes
- Step 6 recorded 94.0% test accuracy; the saved checkpoint measured 88.8% in Step 7 (likely a different run; unverified).
- The Step 6 notebook's tuned custom-CNN size (0.422 M) disagrees with its architecture (0.522 M); the app shows the recomputed value.
- Step 8 results shown are from a stand-in model; re-run Step 8 with the real checkpoint.
