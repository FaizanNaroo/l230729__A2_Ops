# Chapter 7 – API Deployment (Pathway B: Vercel with external inference)

**API URL:** https://l230729-a2-ops.vercel.app
**Endpoints:** `GET /health` (public), `POST /predict` (form-data field `image`, header `X-API-Key`)

## 7.1 Design decision

The brief asks for the model on the cloud and Flask on localhost. I put only the model
behind a small HTTP API on Vercel; the Flask UI and the database stay on my machine
(Docker Compose). Vercel was chosen because it is free, needs no VM administration and
gives an HTTPS URL that is reachable from any network.

Constraints I had to design around:

| Vercel constraint | Consequence | What I did |
|---|---|---|
| Function bundle size limit | TensorFlow does not fit | Converted the champion Keras model to TFLite (`export_tflite.py`) and serve it with the small `ai-edge-litert` runtime |
| No persistent disk | SQLite would be lost on every cold start | The API is stateless; the database stays with Flask |
| ~4.5 MB request body limit | Larger uploads fail before reaching my code | API rejects > 4 MB with a JSON 413; Flask caps uploads at 4 MB in remote mode |
| Cold starts / timeouts | First request is slow | Model is loaded once per instance plus one warm-up inference; `maxDuration: 60`; Flask uses a 30 s timeout |
| Public URL | Anyone could use my model | `/predict` requires an `X-API-Key` header, compared with `hmac.compare_digest` (constant time) |

## 7.2 Keeping training and serving preprocessing identical

TensorFlow is not available on Vercel, so `tf.image.resize` cannot be called. I wrote
`tf_bilinear_resize` in NumPy using the same half-pixel-centre bilinear formula, then
clip and cast to `uint8` exactly as `preprocessing.decode_and_resize` does. The
EfficientNet preprocessing is inside the model graph, so the API feeds raw 0–255
pixels. `verify_tflite.py` checks that Keras and TFLite give the same probability for the
same image. *(Add the output of `python verify_tflite.py` here.)*

## 7.3 Step-by-step deployment (screenshots required for each)

1. `python export_tflite.py` → `vercel_api/model/model.tflite` + `meta.json`. *(Fig 7.1)*
2. `python verify_tflite.py` → Keras vs TFLite probabilities match. *(Fig 7.2)*
3. Install the CLI and log in: `npm i -g vercel`, `vercel login`. *(Fig 7.3)*
4. `cd vercel_api` and `vercel link`, project name `l230729-a2-ops`. *(Fig 7.4)*
5. Add the secret: Vercel dashboard → Project → Settings → Environment Variables →
   `API_KEY` (and optionally `THRESHOLD`). *(Fig 7.5 – value hidden)*
6. Deploy: `vercel --prod`. *(Fig 7.6 – build log and production URL)*
7. `vercel.json` explained: `includeFiles: model/**` ships the model with the function;
   the rewrite sends every path to `api/index.py`, and `VercelPathFix` restores the
   original path so Flask's `/health` and `/predict` routes still match.
8. Check from **another network** (phone hotspot or a friend's PC):
   `curl https://l230729-a2-ops.vercel.app/health`. *(Fig 7.7)*

## 7.4 Postman tests

Collection: `postman/VisionGuard_API.postman_collection.json`. Every request has
assertions; run it with the Collection Runner. *(Fig 7.8 – runner summary; Fig 7.9 – one
successful prediction body)*

| # | Request | Expected | Result |
|---|---|---|---|
| 1 | GET /health | 200, model + threshold | |
| 2 | POST /predict, valid key, real JPG | 200, label/prob_ai/confidence | |
| 3 | POST /predict, no key | 401 | |
| 4 | POST /predict, wrong key | 401 | |
| 5 | POST /predict, text instead of image | 400 | |
| 6 | GET /nope | JSON 404 | |

## 7.5 What went wrong

* Vercel's rewrite turned every request path into `/api/index`, so Flask answered 404 for
  `/predict`. I added a temporary debug route to see the real WSGI path, passed the
  original path as `__path` in the rewrite, restored it in a small WSGI middleware and
  then removed the debug route (commits `e282293`, `bcaec27`, `234873c`).
* *(Add anything else you hit, for example the bundle size error before switching to TFLite.)*

---

# Chapter 8 – Integrating Flask and the API

## 8.1 How it works

`app.py` reads `INFERENCE_MODE` at start-up and builds **one** classifier object:

* `local` → `Classifier` loads the Keras model (Task 4 behaviour);
* `remote` → `RemoteClassifier` (`inference.py`) stores the API URL, key and timeout,
  calls `/health` once to learn the model name and threshold, and reuses one
  `requests.Session` (keep-alive) for all requests.

Both classes have the same `predict(bytes) -> dict` method, so the `/predict` view,
the database code and the templates did not change. The view still validates the
upload locally first (extension, size, real image content), so bad files never reach
the API. Then it forwards the original bytes, measures the round-trip latency and
saves one `Prediction` row: **what** (label, P(AI), threshold), **when** (UTC
timestamp), **by which model** (`visionguard v4 (efficientnet) @ Vercel`), **how
confidently** (confidence, uncertain flag) and the latency.

Error handling. A failure of the API must not look like a failure of the user's image:

| API outcome | Flask response | Stored in DB? |
|---|---|---|
| 200 | redirect to result page | yes |
| 400 / 413 (bad image) | "Upload rejected" page, 400 | no |
| 401 (wrong key) | "Model service unavailable – check API_KEY", 503, logged | no |
| 5xx, timeout, network error | "Model service unavailable", 503, logged | no |

`/health` now also reports `inference.mode`, `api_url` and whether the API is
reachable. Its HTTP code depends only on the database, so a Vercel outage shows as
`"status": "degraded"` without Docker restarting a healthy web container.

Secrets: `API_KEY` is only in the untracked `.env` (passed to the container by Compose)
and in Vercel's environment variables. It is never in Git or in the image.

## 8.2 Evidence (screenshots required)

1. `.env` with `INFERENCE_MODE=remote` and `API_URL` (key blurred). *(Fig 8.1)*
2. `docker compose up -d --build`, then `docker compose ps` shows both services
   healthy. *(Fig 8.2)*
3. `http://localhost:5000/health` showing `"mode": "remote", "reachable": true`. *(Fig 8.3)*
4. Upload at least 4 images in the browser (2 real, 2 AI) and capture each result page.
   The footer reads "Inference: remote API (Vercel)". *(Figs 8.4–8.7)*
5. `python task8_demo.py -n 8`: terminal table of true vs predicted labels and the
   database records read back. *(Fig 8.8)*
6. History page listing all requests with time, verdict, confidence, threshold, model
   and latency. *(Fig 8.9)*
7. Failure case: set a wrong `API_KEY`, restart, upload, and show the 503 page plus the
   log line from `docker compose logs web`. *(Fig 8.10)*

Results from `reports/task8_results.csv`:

| # | True | Predicted | P(AI) | Confidence | Round trip (ms) |
|---|---|---|---|---|---|
| | | | | | |

## 8.3 Discussion

* Latency: the first request after idle includes Vercel's cold start (model load);
  later requests take *(fill in)* ms. Local Keras inference took *(fill in)* ms.
* Remote inference keeps the Flask image free of the model, and lets the model be
  updated by redeploying the API only. The cost is a network dependency, which the
  error pages and `/health` make visible.
