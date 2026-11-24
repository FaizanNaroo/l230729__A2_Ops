# VisionGuard (l230729_A2_Ops)

![CI](https://github.com/FaizanNaroo/l230729__A2_Ops/actions/workflows/ci.yml/badge.svg)

Classifies uploaded images as AI-generated or real.

| | |
|---|---|
| Model API (Vercel) | https://l230729-a2-ops.vercel.app (`GET /health`, `POST /predict`) |
| DagsHub | https://dagshub.com/FaizanNaroo/l230729__A2_Ops |
| Flask UI | runs locally: http://localhost:5000 |

## Architecture

```
Browser ──upload──▶ Flask (localhost:5000) ──HTTPS + X-API-Key──▶ Model API (Vercel, TFLite)
                         │                                              │
                         ▼                                              ▼
                  PostgreSQL / SQLite  ◀── verdict, confidence, model, time, latency
                  (/history, /api/history)
```

* `vercel_api/` – serverless model API. A full TensorFlow install is larger than a
  Vercel function bundle may be, so the champion Keras model is converted to TFLite
  (`export_tflite.py`, checked by `verify_tflite.py`) and served with `ai-edge-litert`.
  Pre-processing re-implements `tf.image.resize` (bilinear, half-pixel) in NumPy so the
  API sees exactly the same pixels as training.
* `app.py` – Flask UI. `INFERENCE_MODE=remote` makes it use `RemoteClassifier`
  (`inference.py`), which forwards the uploaded bytes to the API and stores the answer
  in the database. `INFERENCE_MODE=local` loads the Keras model instead.

## Run

1. Copy `.env.example` to `.env` and fill in the values. `API_KEY` must be identical to
   the `API_KEY` environment variable in the Vercel project.
2. Local Python: `python app.py`. Docker: `docker compose up -d --build`.
3. Open http://localhost:5000, upload an image, then open **History**.

Evidence for Task 8 (several test images through Flask → Vercel → database):

```
python task8_demo.py -n 8          # writes reports/task8_results.csv
python task8_demo.py --direct      # calls the Vercel API directly
```

Postman: import `postman/VisionGuard_API.postman_collection.json`, set the collection
variable `api_key` (current value only), choose an image file in requests 2–4 and run.

## Deploying the model API to Vercel

```
npm i -g vercel
cd vercel_api
vercel login
vercel link                 # project name: l230729-a2-ops
vercel env add API_KEY production      # paste a random secret
vercel --prod
curl https://l230729-a2-ops.vercel.app/health
```

The secret lives only in Vercel's environment variables and the untracked `.env`;
neither the code nor the workflow file contains it.
