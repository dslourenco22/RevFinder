# RevFinder

RevFinder is a local-only Streamlit application for comparing two PDF revisions of engineering documents such as BOMs, purchase orders, and engineering change orders.

The application extracts document text and tables with `pdfplumber`, parses structured line items through a local Ollama model, computes a deterministic revision delta, and exports a styled multi-tab Excel report.

## Security Model

- No external API calls are used.
- The only model endpoint allowed by the parser is `localhost` or `127.0.0.1`.
- PDF bytes, extracted text, parsed fields, and generated reports stay on the internal host.
- Ollama defaults to `http://localhost:11434` with `llama3.2`.

## Directory Layout

```text
RevFinder/
├── app.py
├── requirements.txt
├── README.md
└── src/
    ├── __init__.py
    ├── extractor.py
    ├── llm_parser.py
    ├── engine.py
    └── reporter.py
```

## Setup

Install Python 3.9 or newer, then create a virtual environment:

```bash
cd /opt/RevFinder
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install and run Ollama on the same host:

```bash
ollama pull llama3.2
ollama pull llama3.2:1b
ollama serve
```

## Run Locally

```bash
cd RevFinder
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.address 0.0.0.0 --server.port 8502
```

Open:

```text
http://SERVER_HOSTNAME_OR_IP:8502
```

`requirements.txt` is runtime-only. The model can be switched in the sidebar
(`llama3.2`, `llama3.1:8b`, `qwen2.5:7b`, `llama3.2:1b`); pull any you select with
`ollama pull <model>`. Stronger models map unusual layouts more reliably.

## Testing

```bash
cd RevFinder
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q
```

The suite (`tests/test_revfinder.py`) covers extraction, reconciliation (relational
full outer join with prefix- and content-based key healing), price/quantity
normalization and recovery, severity-coded highlight generation, and the
pipe-delimited log format. CI runs it on every push via `.github/workflows/ci.yml`.

## systemd Deployment

Create `/etc/systemd/system/revfinder.service`:

```ini
[Unit]
Description=RevFinder Streamlit Service
After=network.target ollama.service

[Service]
Type=simple
User=revfinder
Group=revfinder
WorkingDirectory=/opt/RevFinder
Environment=PATH=/opt/RevFinder/.venv/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/opt/RevFinder/.venv/bin/streamlit run app.py --server.address 0.0.0.0 --server.port 8502 --server.headless true
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now revfinder
sudo systemctl status revfinder
```

## Operational Notes

- BoMination can remain on port `8501`; RevFinder binds to `8502`.
- For CPU-only fallback, select `llama3.2:1b` in the sidebar.
- If Ollama is unavailable, RevFinder uses a deterministic table parser and records a parser warning in the UI and Excel summary.
- Accuracy depends on PDF table quality. Native PDFs with selectable text perform best.
