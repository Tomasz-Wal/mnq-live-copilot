import json
import os
import secrets
import sqlite3
import threading
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from openai import OpenAI

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(APP_DIR / "data")))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "mnq_exact_levels.sqlite3"
LATEST_REACTION = DATA_DIR / "latest_reaction.json"

WEBHOOK_TOKEN = os.getenv("WEBHOOK_TOKEN", "change-me")
REPORT_KEY = os.getenv("REPORT_KEY", "change-me-too")
LIVE_MODEL = os.getenv("LIVE_MODEL", "gpt-5.6-luna")
LIVE_REASONING = os.getenv("LIVE_REASONING", "low")
HISTORY_BARS = int(os.getenv("HISTORY_BARS", "18"))  # max 90 min on 5M
REACTION_HISTORY = int(os.getenv("REACTION_HISTORY", "6"))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

app = FastAPI(title="MNQ Live Copilot — Native Indicator Exports", version="4.0.0")
client = OpenAI()
_db_lock = threading.Lock()

LIVE_INSTRUCTIONS = r"""
Jesteś MNQ Live Copilot. Pisz po polsku. Wszystkie godziny podawaj w czasie UK.

BARDZO WAŻNE OGRANICZENIE:
Masz opierać analizę WYŁĄCZNIE na:
1) OHLC zamkniętych świec 5M,
2) dokładnych poziomach przekazanych przez użytkownika z jego wskaźników:
   - Tomasz Range Breakout PRO: Range High / Range Low,
   - Tomasz Volume Profile PRO: VAH / POC / VAL,
   - VWAP z wykresu,
3) historii tych samych danych z bieżącej sesji od 17:30 UK,
4) wcześniejszych reakcji copilota z tej samej sesji.

NIE wolno używać ani wspominać: EMA, RSI, ATR, volume, delta, order flow, news, makro, overnight high/low,
previous day high/low, Initial Balance, własnego ORB, własnego Volume Profile, sigma bands ani jakiegokolwiek poziomu,
którego nie ma w danych wejściowych. Nie wymyślaj brakujących wartości.

Masz oceniać RELACJĘ ceny i zamknięć 5M do dokładnych poziomów użytkownika:
- wybicie / powrót pod Range High albo nad Range Low,
- retest Range High/Low,
- akceptację lub rejection wokół VAH/POC/VAL,
- reclaim / loss / rejection VWAP,
- konfluencję, gdy kilka przekazanych poziomów znajduje się w tym samym rejonie,
- zmianę zachowania względem poprzednich zamkniętych świec 5M.

Jedno zamknięcie po drugiej stronie poziomu traktuj jako wstępne potwierdzenie, a kolejne utrzymanie lub udany retest jako mocniejsze potwierdzenie.
Nie twórz sztucznego setupu, gdy cena jest między poziomami lub zachowanie jest mieszane.
Nie wydawaj polecenia kupna/sprzedaży i nie wykonuj transakcji. Możesz klasyfikować środowisko i opisywać warunki do obserwacji.

Odpowiedź ma być krótka i użyteczna:
[HH:MM UK] LONG ENV / SHORT ENV / NEUTRAL — X/10
Zmiana 5M: co dokładnie zmieniło się względem poprzedniej świecy/reakcji.
Range: reakcja względem Range High/Low.
VP: reakcja względem VAH/POC/VAL.
VWAP: reclaim / loss / hold / rejection.
Konfluencja: maks. jedno zdanie, tylko jeśli faktycznie istnieje.
Teraz obserwuj: maksymalnie 2 konkretne warunki oparte wyłącznie na przekazanych poziomach.
Stan: WAIT / WATCH LONG / WATCH SHORT / MIXED.

Jeśli od poprzednich 5 minut nic ważnego się nie zmieniło, napisz to wprost i nie podnoś oceny bez powodu.
""".strip()

REPORT_INSTRUCTIONS = r"""
Przygotuj zwięzłe podsumowanie bieżącej sesji MNQ od 17:30 UK.
Używaj WYŁĄCZNIE OHLC 5M oraz dokładnych poziomów Range High/Low, VP VAH/POC/VAL i VWAP.
Nie dodawaj żadnych innych wskaźników, poziomów, newsów ani makro.

Struktura:
A. Stan sesji w 3 zdaniach
B. Range Breakout — co cena zrobiła przy Range High/Low
C. Volume Profile — VAH/POC/VAL i acceptance/rejection
D. VWAP — relacja i najważniejsze reclaim/loss/rejection
E. Aktualne środowisko LONG/SHORT/NEUTRAL 0-10
F. 2 scenariusze warunkowe na kolejne świece 5M, wyłącznie względem przekazanych poziomów
""".strip()


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db() -> None:
    with _db_lock, db() as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                received_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                market_time_uk TEXT NOT NULL,
                event TEXT NOT NULL,
                symbol TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_id INTEGER NOT NULL,
                generated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                model TEXT NOT NULL,
                reaction TEXT NOT NULL,
                FOREIGN KEY(snapshot_id) REFERENCES snapshots(id)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_snapshot
                ON snapshots(symbol, market_time_uk, event);
            CREATE INDEX IF NOT EXISTS idx_market_time ON snapshots(market_time_uk);
            CREATE INDEX IF NOT EXISTS idx_reaction_snapshot ON reactions(snapshot_id);
            """
        )


init_db()


def parse_market_time(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M")


def in_allowed_window(value: str) -> bool:
    try:
        dt = parse_market_time(value)
    except Exception:
        return False
    mins = dt.hour * 60 + dt.minute
    return (17 * 60 + 30) <= mins <= (22 * 60)


def insert_snapshot(payload: dict[str, Any]) -> tuple[int, bool]:
    market_time = str(payload.get("time_uk") or "")
    event = str(payload.get("event") or "")
    symbol = str(payload.get("symbol") or "")
    body = json.dumps(payload, ensure_ascii=False)

    with _db_lock, db() as con:
        cur = con.execute(
            "INSERT OR IGNORE INTO snapshots(market_time_uk,event,symbol,payload_json) VALUES(?,?,?,?)",
            (market_time, event, symbol, body),
        )
        if cur.rowcount == 1:
            return int(cur.lastrowid), True
        row = con.execute(
            "SELECT id FROM snapshots WHERE symbol=? AND market_time_uk=? AND event=?",
            (symbol, market_time, event),
        ).fetchone()
        if not row:
            raise RuntimeError("Could not retrieve duplicate snapshot")
        return int(row["id"]), False


def get_snapshot(snapshot_id: int) -> dict[str, Any]:
    with _db_lock, db() as con:
        row = con.execute("SELECT payload_json FROM snapshots WHERE id=?", (snapshot_id,)).fetchone()
    if not row:
        raise KeyError(snapshot_id)
    payload = json.loads(row["payload_json"])
    payload["_snapshot_id"] = snapshot_id
    return payload


def session_snapshots(snapshot_id: int, limit: int = HISTORY_BARS) -> list[dict[str, Any]]:
    current = get_snapshot(snapshot_id)
    day = str(current["time_uk"])[:10]
    with _db_lock, db() as con:
        rows = con.execute(
            """
            SELECT id, payload_json FROM snapshots
            WHERE market_time_uk LIKE ?
            ORDER BY id DESC LIMIT ?
            """,
            (day + "%", limit),
        ).fetchall()
    result: list[dict[str, Any]] = []
    for row in reversed(rows):
        p = json.loads(row["payload_json"])
        p["_snapshot_id"] = row["id"]
        result.append(p)
    return result


def session_reactions(snapshot_id: int, limit: int = REACTION_HISTORY) -> list[dict[str, Any]]:
    current = get_snapshot(snapshot_id)
    day = str(current["time_uk"])[:10]
    with _db_lock, db() as con:
        rows = con.execute(
            """
            SELECT r.id, r.snapshot_id, r.generated_at, r.model, r.reaction, s.market_time_uk
            FROM reactions r
            JOIN snapshots s ON s.id=r.snapshot_id
            WHERE s.market_time_uk LIKE ?
            ORDER BY r.id DESC LIMIT ?
            """,
            (day + "%", limit),
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def insert_reaction(snapshot_id: int, model: str, reaction: str) -> None:
    with _db_lock, db() as con:
        con.execute(
            "INSERT INTO reactions(snapshot_id,model,reaction) VALUES(?,?,?)",
            (snapshot_id, model, reaction),
        )


def send_telegram(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": TELEGRAM_CHAT_ID, "text": text}).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=7):
        pass


def send_discord(text: str) -> None:
    if not DISCORD_WEBHOOK_URL:
        return
    body = json.dumps({"content": text[:1900]}).encode("utf-8")
    req = urllib.request.Request(
        DISCORD_WEBHOOK_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=7):
        pass


def push_reaction(text: str) -> None:
    for fn in (send_telegram, send_discord):
        try:
            fn(text)
        except Exception:
            pass


def analyse_live(snapshot_id: int) -> None:
    try:
        history = session_snapshots(snapshot_id, HISTORY_BARS)
        reactions = session_reactions(snapshot_id, REACTION_HISTORY)
        current = history[-1]

        prompt = (
            "BIEŻĄCY SNAPSHOT:\n"
            + json.dumps(current, ensure_ascii=False, indent=2)
            + "\n\nHISTORIA TEJ SAMEJ SESJI 5M (najstarsza -> najnowsza):\n"
            + json.dumps(history, ensure_ascii=False, indent=2)
            + "\n\nOSTATNIE REAKCJE Z TEJ SAMEJ SESJI:\n"
            + json.dumps(reactions, ensure_ascii=False, indent=2)
        )

        resp = client.responses.create(
            model=LIVE_MODEL,
            instructions=LIVE_INSTRUCTIONS,
            input=prompt,
            reasoning={"effort": LIVE_REASONING},
        )
        reaction = resp.output_text.strip()
        insert_reaction(snapshot_id, LIVE_MODEL, reaction)
        latest = {
            "snapshot_id": snapshot_id,
            "market_time_uk": current.get("time_uk"),
            "model": LIVE_MODEL,
            "reaction": reaction,
        }
        atomic_write(LATEST_REACTION, latest)
        push_reaction(reaction)
    except Exception as exc:
        atomic_write(
            LATEST_REACTION,
            {"snapshot_id": snapshot_id, "error": f"{type(exc).__name__}: {exc}"},
        )


def build_report() -> str:
    with _db_lock, db() as con:
        last = con.execute("SELECT id FROM snapshots ORDER BY id DESC LIMIT 1").fetchone()
    if not last:
        return "Brak snapshotów z bieżącej sesji."
    sid = int(last["id"])
    history = session_snapshots(sid, 100)
    reactions = session_reactions(sid, 100)
    resp = client.responses.create(
        model=LIVE_MODEL,
        instructions=REPORT_INSTRUCTIONS,
        input=(
            "SNAPSHOTY SESJI:\n"
            + json.dumps(history, ensure_ascii=False, indent=2)
            + "\n\nREAKCJE LIVE:\n"
            + json.dumps(reactions, ensure_ascii=False, indent=2)
        ),
        reasoning={"effort": "medium"},
    )
    return resp.output_text.strip()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model": LIVE_MODEL,
        "window_uk": "17:30-22:00",
        "history_bars": HISTORY_BARS,
        "telegram": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
        "discord": bool(DISCORD_WEBHOOK_URL),
        "data_policy": "OHLC 5M + exact external indicator levels + VWAP only",
    }


@app.post("/tv/{token}", status_code=202)
async def tradingview_webhook(token: str, request: Request, background_tasks: BackgroundTasks) -> dict[str, Any]:
    if not secrets.compare_digest(token, WEBHOOK_TOKEN):
        raise HTTPException(status_code=403, detail="Invalid webhook token")
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Expected valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")

    required = {"event", "symbol", "time_uk", "ohlc", "levels"}
    if not required.issubset(payload):
        raise HTTPException(status_code=400, detail=f"Missing fields: {sorted(required - set(payload))}")
    if payload.get("event") != "mnq_bar_close_5m_exact_levels":
        raise HTTPException(status_code=400, detail="Unexpected event")
    if not in_allowed_window(str(payload.get("time_uk"))):
        return {"accepted": True, "ignored": True, "reason": "outside 17:30-22:00 UK"}

    snapshot_id, is_new = insert_snapshot(payload)
    if is_new:
        background_tasks.add_task(analyse_live, snapshot_id)
    return {"accepted": True, "snapshot_id": snapshot_id, "new": is_new}


@app.get("/latest/{key}")
def latest_json(key: str) -> dict[str, Any]:
    if not secrets.compare_digest(key, REPORT_KEY):
        raise HTTPException(status_code=403, detail="Invalid report key")
    if not LATEST_REACTION.exists():
        raise HTTPException(status_code=404, detail="No reaction generated yet")
    return json.loads(LATEST_REACTION.read_text(encoding="utf-8"))


@app.get("/reaction/{key}", response_class=PlainTextResponse)
def latest_reaction(key: str) -> str:
    data = latest_json(key)
    return data.get("reaction") or data.get("error", "No reaction")


@app.get("/feed/{key}")
def feed(key: str, limit: int = 40) -> dict[str, Any]:
    if not secrets.compare_digest(key, REPORT_KEY):
        raise HTTPException(status_code=403, detail="Invalid report key")
    limit = max(1, min(limit, 100))
    with _db_lock, db() as con:
        rows = con.execute(
            """
            SELECT r.id, r.snapshot_id, r.generated_at, r.model, r.reaction, s.market_time_uk
            FROM reactions r JOIN snapshots s ON s.id=r.snapshot_id
            ORDER BY r.id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return {"items": [dict(r) for r in reversed(rows)]}


@app.get("/report/{key}", response_class=PlainTextResponse)
def report(key: str) -> str:
    if not secrets.compare_digest(key, REPORT_KEY):
        raise HTTPException(status_code=403, detail="Invalid report key")
    return build_report()


@app.get("/dashboard/{key}", response_class=HTMLResponse)
def dashboard(key: str) -> str:
    if not secrets.compare_digest(key, REPORT_KEY):
        raise HTTPException(status_code=403, detail="Invalid report key")
    safe_key = json.dumps(key)
    return f"""
<!doctype html>
<html lang="pl">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>MNQ Exact Levels Copilot</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:0;background:#0b0d10;color:#e9eef5}}
.wrap{{max-width:900px;margin:auto;padding:24px}}
h1{{font-size:24px;margin:0 0 8px}}
.sub{{color:#9aa6b2;margin-bottom:18px}}
.card{{background:#151a20;border:1px solid #28313b;border-radius:14px;padding:16px;margin:12px 0;white-space:pre-wrap;line-height:1.45}}
.time{{font-weight:700;margin-bottom:8px}}
</style>
</head>
<body><div class="wrap">
<h1>MNQ Live Copilot — Exact Levels</h1>
<div class="sub">Tylko Twoje Range / VP / VWAP + zamknięte świece 5M. 17:30–22:00 UK.</div>
<div id="feed"></div></div>
<script>
const key={safe_key};
function esc(s){{return String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;')}}
async function load(){{
  const r=await fetch(`/feed/${{encodeURIComponent(key)}}?limit=40`,{{cache:'no-store'}});
  const j=await r.json();
  const root=document.getElementById('feed'); root.innerHTML='';
  [...j.items].reverse().forEach(x=>{{
    const d=document.createElement('div'); d.className='card';
    d.innerHTML=`<div class="time">${{esc(x.market_time_uk || '')}}</div>${{esc(x.reaction)}}`;
    root.appendChild(d);
  }});
}}
load(); setInterval(load,5000);
</script>
</body></html>
"""
