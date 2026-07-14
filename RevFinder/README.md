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

## Windows LAN Deployment (host on a Windows "server")

RevFinder runs great on a Windows laptop/desktop that stays on the LAN. Ollama is
**optional** — structured PO/BoM/Excel/CSV documents are parsed by the built-in
deterministic engine, so you only need Ollama for unusual PDF layouts.

1. **Install Python** 3.9+ from <https://www.python.org/downloads/> — during setup
   check **"Add python.exe to PATH"**.

2. **Copy the app folder** (the `RevFinder` directory containing `app.py`,
   `requirements.txt`, `src/`, `.streamlit/`) to the Windows machine, e.g.
   `C:\RevFinder`. Do not copy the `.venv` folder — it is recreated on first run.
   (Or `git clone` the repo if you use one.)

3. **Run it** — double-click `run.bat` (or from a command prompt: `run.bat`). The
   first run creates a virtual environment and installs dependencies, then serves
   on port `8502`.

4. **Open the Windows Firewall** for the port (one-time, in an **Administrator**
   command prompt):

   ```bat
   netsh advfirewall firewall add rule name="RevFinder 8502" dir=in action=allow protocol=TCP localport=8502
   ```

5. **Find the machine's IP** with `ipconfig` (the IPv4 Address). Team members open:

   ```text
   http://THAT-IP:8502
   ```

### Auto-start on boot (keep it always running)

Easiest — **Task Scheduler**:

1. Open *Task Scheduler* → *Create Task*.
2. General: *Run whether user is logged on or not*, *Run with highest privileges*.
3. Triggers: *At startup*.
4. Actions: *Start a program* → Program/script: `C:\RevFinder\run.bat` (Start in:
   `C:\RevFinder`).
5. Save. It now launches on boot and restarts with the machine.

More robust (auto-restart on crash) — install [NSSM](https://nssm.cc/) and register
`run.bat` (or the venv `streamlit.exe`) as a Windows service.

## systemd Deployment (Linux)

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
