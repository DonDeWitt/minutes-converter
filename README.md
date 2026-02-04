# minutes-converter# Motorcycle Club Minutes Converter

Convert decades of motorcycle club meeting minutes from raw text to standardized JSONL for archival and search.

## Features
- Splits a large text file of meeting minutes into individual meetings
- Uses Gemini 1.5 Flash (via LangChain) to extract structured data
- Validates output with Pydantic
- Appends results to a JSONL file (one object per line)
- Logs errors and skips malformed entries
- Respects Google AI Studio free-tier rate limits

## Usage
1. Copy `.env.example` to `.env` and add your Google API key.
2. Place your minutes in `minutes.txt` (use `***` or `---` as meeting separators).
3. Build the dev container or install requirements:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the script:
   ```bash
   python minutes_to_jsonl.py
   ```
5. Output will be in `formatted_minutes.jsonl`. Errors are logged in `errors.log`.

## Environment Variables
- `GOOGLE_API_KEY`: Your Google AI Studio API key (see `.env.example`).

## Dev Container
A dev container is provided for reproducible setup. See `.devcontainer/`.
