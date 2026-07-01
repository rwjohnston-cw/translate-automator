# Vocal Score Translator

Vocal Score Translator is a Dockerised FastAPI web app that accepts a vocal/choral score PDF, sends rendered page images to an LLM provider for lyric recognition + translation, and generates a downloadable PDF that preserves original pages while overlaying translated text.

## Architecture

- **Web layer**: FastAPI + Jinja2 templates + vanilla JS/CSS.
- **Upload + validation**: strict PDF checks (extension, signature, readable by PyMuPDF, no password, size/page limits).
- **Job system**: filesystem job manifests at `/tmp/score-translator/jobs/<uuid>/` with status polling and TTL cleanup.
- **Rendering for model analysis**: each source page rendered to PNG (white background, DPI + max-dimension constrained) in page order.
- **LLM analysis**: context-aware batching with owned pages and context-only pages using OpenAI, Gemini API, or DeepSeek (testing mode) with Pydantic-validated structured outputs.
- **Output PDF**: original pages are kept intact; translated overlays are inserted with `insert_htmlbox()` and Noto font fallbacks.

## Local Docker Setup

```bash
cp .env.example .env
# add OPENAI_API_KEY to .env
docker compose up --build
```

Open: [http://localhost:8000](http://localhost:8000)

To follow processing logs live:

```bash
docker compose logs -f web
```

## Environment Variables

Defined in `.env.example`:

- `OPENAI_API_KEY`
- `OPENAI_MODEL` (default `gpt-5.5`)
- `OPENAI_REASONING_EFFORT` (`low|medium|high`)
- `GEMINI_API_KEY`
- `GEMINI_MODEL`
- `DEEPSEEK_API_KEY`
- `DEEPSEEK_MODEL`
- `DEEPSEEK_BASE_URL`
- `MAX_UPLOAD_MB`
- `MAX_PAGES`
- `OWNED_BATCH_SIZE`
- `CONTEXT_PAGES`
- `MAX_PARALLEL_BATCHES` (default `2`)
- `IMAGE_DPI`
- `IMAGE_MAX_DIMENSION`
- `JOB_TTL_MINUTES`
- `OUTPUT_FONT_SIZE`
- `OUTPUT_BACKGROUND_OPACITY`

## How Batching Works

- Every page belongs to exactly one **owned batch** (default size 6).
- Up to `CONTEXT_PAGES` before/after are included for continuity.
- Batch requests are processed concurrently with a cap controlled by `MAX_PARALLEL_BATCHES`.
- The prompt explicitly names:
  - supplied range
  - owned range
  - context-only pages
- Model outputs outside owned range are filtered.
- Batch results are merged, deduplicated (exact page+position+text), and sorted by page then configured position order.

## How Page Positioning Works

Default positioning uses:

- `top`: x 5–95%, y 2–13%
- `middle`: x 5–95%, y 44–56%
- `bottom`: x 5–95%, y 87–98%

Testing mode also supports:

- 5-point centered regions (`top`, `upper_middle`, `middle`, `lower_middle`, `bottom`)
- 6-point split regions (`top_left`, `top_right`, `middle_left`, `middle_right`, `bottom_left`, `bottom_right`)
- 20-point grid regions (`a1-a4`, `b1-b4`, `c1-c4`, `d1-d4`, `e1-e5`)

Multiple fragments in the same region are joined with line breaks and rendered using HTML/CSS with escaped text.

## Testing Mode (UI)

Enable **Testing mode** on the upload page to expose:

- LLM provider selector (OpenAI, Gemini API, DeepSeek)
- Model selector/custom model override
- Positioning variant selector (3-point, 5-point centered, 6-point split, 20-point grid)
- Batching overrides for `owned_batch_size` and `context_pages`

## Job Logs

When a job completes, the status page provides a **Download LLM request log (JSON)** link.
The log includes per-batch and total telemetry:

- Provider/model used and reasoning effort
- Pages sent to each request
- Request duration
- Input/output/total tokens
- Estimated USD cost using model pricing tables
- Prompt snapshot (images redacted as placeholders)
- Structured information returned by the model

## Privacy

- API key remains server-side only.
- The browser never receives the API key.
- The server does **not** log API keys, base64 image payloads, or full OpenAI request payloads.
- Temporary files are stored per job and cleaned up after TTL expiry.
- Rendered page images are sent to OpenAI for recognition and translation.

## Running Tests

```bash
python -m pip install -r requirements.txt
pytest
```

Tests mock provider interactions and do not call real external APIs.

## Troubleshooting

- **Upload rejected**: confirm it is a valid PDF and within `MAX_UPLOAD_MB` and `MAX_PAGES`.
- **Job fails early**: verify `OPENAI_API_KEY` is set in `.env`.
- **Font/script rendering issues**: ensure Docker image includes Noto font packages (already configured in `Dockerfile`).
- **404 on download**: job may be expired by TTL cleanup or still in progress.

## Known Limitations

- Very low-resolution or handwritten scores may be difficult to recognise.
- Highly polytextual music may require manual checking.
- The model may occasionally divide a continuous phrase imperfectly between page regions.
- Translation boxes can overlap dense score content because the requested placement system is limited to top, middle, and bottom.
- API use incurs OpenAI charges.
- Output should be reviewed before publication or performance use.

