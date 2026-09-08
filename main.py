import json
import os
import secrets
import sqlite3
import threading
import urllib.parse
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo
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


UK_TZ = ZoneInfo("Europe/London")

def parse_market_time(value: str) -> datetime:
    value = value.strip()

    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))

        if dt.tzinfo is not None:
            return dt.astimezone(UK_TZ).replace(tzinfo=None)
    except ValueError:
        pass

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
        current_for_ai = dict(current)

        if current.get("time_uk"):
            dt_utc = datetime.fromisoformat(current["time_uk"].replace("Z", "+00:00"))
            dt_uk = dt_utc.astimezone(UK_TZ)
            current_for_ai["time_uk"] = dt_uk.strftime("%Y-%m-%d %H:%M UK")

        prompt = (
            "BIEŻĄCY SNAPSHOT:\n"
            + json.dumps(current_for_ai, ensure_ascii=False, indent=2)
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

        print("TV PAYLOAD:", json.dumps(payload, ensure_ascii=False), flush=True)
        print(
            "TIME_UK:",
            payload.get("time_uk"),
            "ALLOWED:",
            in_allowed_window(str(payload.get("time_uk"))),
            flush=True,
)
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


@app.get("/latest/{key:path}")
def latest_json(key: str) -> dict[str, Any]:
    if not secrets.compare_digest(key, REPORT_KEY):
        raise HTTPException(status_code=403, detail="Invalid report key")
    if not LATEST_REACTION.exists():
        raise HTTPException(status_code=404, detail="No reaction generated yet")
    return json.loads(LATEST_REACTION.read_text(encoding="utf-8"))


@app.get("/reaction/{key:path}", response_class=PlainTextResponse)
def latest_reaction(key: str) -> str:
    data = latest_json(key)
    return data.get("reaction") or data.get("error", "No reaction")


@app.get("/feed/{key:path}")
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


@app.get("/report/{key:path}", response_class=PlainTextResponse)
def report(key: str) -> str:
    if not secrets.compare_digest(key, REPORT_KEY):
        raise HTTPException(status_code=403, detail="Invalid report key")
    return build_report()


@app.get("/dashboard/{key:path}", response_class=HTMLResponse)
def dashboard(key: str) -> str:
    if not secrets.compare_digest(key, REPORT_KEY):
        raise HTTPException(status_code=403, detail="Invalid report key")

    safe_key = json.dumps(key)

    html = """
<!doctype html>
<html lang="pl">

<head>

<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>

<title>MNQ Live Copilot</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    background: #080b0f;
    color: #eaf0f6;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.app {
    max-width: 1100px;
    margin: auto;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
}

/* ============================= */
/* HEADER */
/* ============================= */

.header {
    position: sticky;
    top: 0;
    z-index: 10;

    padding: 18px 22px;

    background: rgba(8,11,15,.96);
    backdrop-filter: blur(12px);

    border-bottom: 1px solid #202832;
}

.header-top {
    display: flex;
    align-items: center;
    gap: 20px;
}

.title-area {
    min-width: 260px;
}

.title {
    font-size: 22px;
    font-weight: 750;
}

.subtitle {
    margin-top: 5px;
    color: #8997a7;
    font-size: 13px;
}

/* ============================= */
/* ACCOUNT */
/* ============================= */

.account-box {
    margin-left: auto;

    display: flex;
    align-items: center;

    gap: 18px;

    padding: 8px 12px;

    border: 1px solid #26303a;
    border-radius: 12px;

    background: #11171d;
}

.account-row {
    display: flex;
    flex-direction: column;
    gap: 2px;

    min-width: 100px;
}

.account-label {
    font-size: 10px;
    color: #7f8b97;
    font-weight: 700;
}

.account-value {
    font-size: 15px;
    font-weight: 750;
}

.pnl-positive {
    color: #49dc87;
}

.pnl-negative {
    color: #ff7474;
}

.pnl-zero {
    color: #aab4be;
}

.account-edit {
    border: 1px solid #33404c;

    background: #192129;
    color: #dbe4ec;

    border-radius: 8px;

    padding: 6px 10px;

    cursor: pointer;
}

.account-edit:hover {
    background: #222c36;
}

/* ============================= */
/* LIVE */
/* ============================= */

.live {
    display: flex;
    align-items: center;

    gap: 7px;

    font-size: 11px;
    color: #a9b4bf;
}

.dot {
    width: 8px;
    height: 8px;

    border-radius: 50%;

    background: #43d17a;

    box-shadow: 0 0 8px #43d17a;
}

/* ============================= */
/* FEED */
/* ============================= */

.feed {
    flex: 1;

    padding: 24px 22px 80px;
}

.message {
    margin-bottom: 18px;

    animation: appear .25s ease;
}

@keyframes appear {

    from {
        opacity: 0;
        transform: translateY(-7px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }

}

/* ============================= */
/* MESSAGE HEADER */
/* ============================= */

.meta {
    display: flex;
    align-items: center;

    gap: 9px;

    margin-bottom: 7px;

    font-size: 12px;
    color: #83909d;
}

.time {
    font-weight: 650;
    color: #b6c0cb;
}

.badge {
    padding: 3px 8px;

    border-radius: 999px;

    font-size: 11px;
    font-weight: 750;
}

.score {
    margin-left: auto;

    display: flex;
    align-items: center;
    gap: 7px;

    font-size: 12px;
    font-weight: 700;
}

.delta-up {
    color: #53db87;
}

.delta-down {
    color: #ff7070;
}

.delta-flat {
    color: #8997a7;
}

/* ============================= */
/* MESSAGE BODY */
/* ============================= */

.bubble {
    background: #12171d;

    border: 1px solid #252e38;

    border-radius: 15px;

    padding: 15px 17px;

    line-height: 1.55;

    font-size: 14px;
}

/* LONG */

.message.long .bubble {
    border-left: 4px solid #3bd67f;
}

.env-long {
    color: #49dc87;

    background: rgba(73,220,135,.12);
}

/* SHORT */

.message.short .bubble {
    border-left: 4px solid #ff6262;
}

.env-short {
    color: #ff7474;

    background: rgba(255,116,116,.12);
}

/* NEUTRAL */

.message.neutral .bubble {
    border-left: 4px solid #e0b64d;
}

.env-neutral {
    color: #e6bd58;

    background: rgba(230,189,88,.12);
}

/* STATE */

.state {
    margin-left: 3px;

    background: #202832;

    color: #cbd5df;
}

.state-long {
    background: rgba(73,220,135,.13);

    color: #55df8e;
}

.state-short {
    background: rgba(255,116,116,.13);

    color: #ff7777;
}

/* ============================= */
/* ANALYSIS TEXT */
/* ============================= */

.section {
    margin-top: 8px;
}

.section:first-child {
    margin-top: 0;
}

.label {
    font-weight: 750;

    color: #d9e2eb;
}

.watch {
    margin-top: 11px;

    padding-top: 11px;

    border-top: 1px solid #252d36;
}

/* ============================= */
/* EMPTY */
/* ============================= */

.empty {
    text-align: center;

    margin-top: 100px;

    color: #71808e;
}

/* ============================= */
/* STATUS */
/* ============================= */

.footer-status {
    position: fixed;

    bottom: 15px;

    left: 50%;

    transform: translateX(-50%);

    padding: 7px 13px;

    background: #151b21;

    border: 1px solid #29323c;

    border-radius: 999px;

    color: #81909e;

    font-size: 11px;

    box-shadow: 0 5px 20px rgba(0,0,0,.35);
}

/* ============================= */
/* MOBILE */
/* ============================= */

@media(max-width:800px) {

    .header {
        padding: 14px;
    }

    .header-top {
        flex-wrap: wrap;
    }

    .account-box {
        width: 100%;
        margin-left: 0;

        justify-content: space-between;
    }

    .feed {
        padding: 18px 12px 70px;
    }

    .bubble {
        font-size: 13px;
    }

    .subtitle {
        font-size: 12px;
    }

}

</style>

</head>

<body>

<div class="app">

<header class="header">

<div class="header-top">

<div class="title-area">

<div class="title">
MNQ Live Copilot
</div>

<div class="subtitle">
Range / Volume Profile / VWAP · 5M · 17:30–22:00 UK
</div>

</div>


<div class="account-box">

<div class="account-row">

<span class="account-label">
BALANCE
</span>

<span id="balanceValue" class="account-value">
£0.00
</span>

</div>


<div class="account-row">

<span class="account-label">
TODAY PNL
</span>

<span id="pnlValue" class="account-value pnl-zero">
£0.00
</span>

</div>


<button
class="account-edit"
onclick="editAccount()"
>
Edit
</button>

</div>


<div class="live">

<span class="dot"></span>

LIVE

</div>

</div>

</header>


<main id="feed" class="feed">

<div class="empty">
Oczekiwanie na analizę...
</div>

</main>

</div>


<div id="status" class="footer-status">
Łączenie...
</div>


<script>

const key = __SAFE_KEY__;

let lastIds = new Set();

let previousScore = null;

let firstLoad = true;


/* ================================= */
/* SECURITY */
/* ================================= */

function esc(value) {

    return String(value ?? "")

        .replaceAll("&", "&amp;")

        .replaceAll("<", "&lt;")

        .replaceAll(">", "&gt;")

        .replaceAll('"', "&quot;");

}


/* ================================= */
/* UK TIME */
/* ================================= */

function ukTime(raw) {

    if (!raw)
        return "";

    try {

        const date = new Date(raw);

        return new Intl.DateTimeFormat(

            "en-GB",

            {

                timeZone: "Europe/London",

                hour: "2-digit",

                minute: "2-digit",

                hour12: false

            }

        ).format(date) + " UK";

    }

    catch(e) {

        return raw;

    }

}


/* ================================= */
/* ACCOUNT */
/* ================================= */

function loadAccount() {

    const balance =
        localStorage.getItem("mnq_balance") || "0";

    const pnl =
        localStorage.getItem("mnq_pnl") || "0";

    updateAccountDisplay(
        balance,
        pnl
    );

}


function updateAccountDisplay(balance, pnl) {

    const balanceEl =
        document.getElementById("balanceValue");

    const pnlEl =
        document.getElementById("pnlValue");


    let balanceNum =
        Number(balance);

    let pnlNum =
        Number(pnl);


    if (!Number.isFinite(balanceNum))
        balanceNum = 0;

    if (!Number.isFinite(pnlNum))
        pnlNum = 0;


    balanceEl.textContent =
        "£" + balanceNum.toFixed(2);


    pnlEl.classList.remove(

        "pnl-positive",

        "pnl-negative",

        "pnl-zero"

    );


    if (pnlNum > 0) {

        pnlEl.classList.add(
            "pnl-positive"
        );

        pnlEl.textContent =
            "+£" + pnlNum.toFixed(2);

    }

    else if (pnlNum < 0) {

        pnlEl.classList.add(
            "pnl-negative"
        );

        pnlEl.textContent =
            "-£" + Math.abs(pnlNum).toFixed(2);

    }

    else {

        pnlEl.classList.add(
            "pnl-zero"
        );

        pnlEl.textContent =
            "£0.00";

    }

}


function editAccount() {

    const currentBalance =
        localStorage.getItem("mnq_balance") || "0";

    const currentPnl =
        localStorage.getItem("mnq_pnl") || "0";


    const balance = prompt(

        "Podaj aktualne saldo:",

        currentBalance

    );


    if (balance === null)
        return;


    const pnl = prompt(

        "Podaj dzisiejszy PnL:",

        currentPnl

    );


    if (pnl === null)
        return;


    const balanceNumber =
        Number(
            String(balance)
                .replace(",", ".")
        );


    const pnlNumber =
        Number(
            String(pnl)
                .replace(",", ".")
        );


    if (!Number.isFinite(balanceNumber)) {

        alert("Nieprawidłowe saldo.");

        return;

    }


    if (!Number.isFinite(pnlNumber)) {

        alert("Nieprawidłowy PnL.");

        return;

    }


    localStorage.setItem(

        "mnq_balance",

        String(balanceNumber)

    );


    localStorage.setItem(

        "mnq_pnl",

        String(pnlNumber)

    );


    loadAccount();

}


/* ================================= */
/* ANALYSIS PARSER */
/* ================================= */

function parseReaction(text) {

    text =
        String(text || "");


    let environment =
        "NEUTRAL";


    if (/LONG ENV/i.test(text))

        environment =
            "LONG ENV";


    else if (/SHORT ENV/i.test(text))

        environment =
            "SHORT ENV";


    const scoreMatch =

        text.match(

            /(?:LONG ENV|SHORT ENV|NEUTRAL)\\s*[—-]\\s*(\\d+)\\s*\\/\\s*10/i

        );


    const score =

        scoreMatch

        ? Number(scoreMatch[1])

        : null;


    const stateMatch =

        text.match(

            /Stan:\\s*([^\\n]+)/i

        );


    const state =

        stateMatch

        ? stateMatch[1].trim()

        : "";


    return {

        environment,

        score,

        state

    };

}


/* ================================= */
/* FORMAT ANALYSIS */
/* ================================= */

function formatReaction(text) {

    let lines =
        String(text || "").split("\\n");


    if (

        lines.length &&

        /^\\s*\\[.*UK\\].*(LONG ENV|SHORT ENV|NEUTRAL)/i.test(
            lines[0]
        )

    ) {

        lines.shift();

    }


    return lines.map(line => {

        const safe =
            esc(line);


        if (/^Zmiana 5M:/i.test(line))

            return `

            <div class="section">

            <span class="label">
            Zmiana 5M:
            </span>

            ${safe.substring(
                safe.indexOf(":") + 1
            )}

            </div>

            `;


        if (/^Range:/i.test(line))

            return `

            <div class="section">

            <span class="label">
            Range:
            </span>

            ${safe.substring(
                safe.indexOf(":") + 1
            )}

            </div>

            `;


        if (/^VP:/i.test(line))

            return `

            <div class="section">

            <span class="label">
            VP:
            </span>

            ${safe.substring(
                safe.indexOf(":") + 1
            )}

            </div>

            `;


        if (/^VWAP:/i.test(line))

            return `

            <div class="section">

            <span class="label">
            VWAP:
            </span>

            ${safe.substring(
                safe.indexOf(":") + 1
            )}

            </div>

            `;


        if (/^Konfluencja:/i.test(line))

            return `

            <div class="section">

            <span class="label">
            Konfluencja:
            </span>

            ${safe.substring(
                safe.indexOf(":") + 1
            )}

            </div>

            `;


        if (/^Teraz obserwuj:/i.test(line))

            return `

            <div class="section watch">

            <span class="label">
            Teraz obserwuj:
            </span>

            </div>

            `;


        if (/^Stan:/i.test(line))

            return "";


        if (/^\\s*[-•]/.test(line))

            return `

            <div class="section">

            ${safe}

            </div>

            `;


        if (!line.trim())

            return "";


        return `

        <div class="section">

        ${safe}

        </div>

        `;

    }).join("");

}


/* ================================= */
/* CREATE MESSAGE */
/* ================================= */

function createMessage(item) {

    const parsed =
        parseReaction(
            item.reaction
        );


    const wrapper =
        document.createElement("div");


    wrapper.className =
        "message";


    let envClass =
        "neutral";


    let envBadge =
        "env-neutral";


    if (
        parsed.environment ===
        "LONG ENV"
    ) {

        envClass =
            "long";

        envBadge =
            "env-long";

    }


    else if (
        parsed.environment ===
        "SHORT ENV"
    ) {

        envClass =
            "short";

        envBadge =
            "env-short";

    }


    wrapper.classList.add(
        envClass
    );


    let deltaHTML =
        "";


    if (

        parsed.score !== null &&

        previousScore !== null

    ) {

        const delta =

            parsed.score -
            previousScore;


        if (delta > 0) {

            deltaHTML =

                `<span class="delta-up">
                ▲ +${delta}
                </span>`;

        }


        else if (delta < 0) {

            deltaHTML =

                `<span class="delta-down">
                ▼ ${delta}
                </span>`;

        }


        else {

            deltaHTML =

                `<span class="delta-flat">
                → 0
                </span>`;

        }

    }


    if (parsed.score !== null)

        previousScore =
            parsed.score;


    let stateClass =
        "state";


    if (
        /WATCH LONG/i.test(
            parsed.state
        )
    )

        stateClass +=
            " state-long";


    if (
        /WATCH SHORT/i.test(
            parsed.state
        )
    )

        stateClass +=
            " state-short";


    wrapper.innerHTML = `

    <div class="meta">

        <span class="time">

        ${esc(
            ukTime(
                item.market_time_uk
            )
        )}

        </span>


        <span class="badge ${envBadge}">

        ${esc(
            parsed.environment
        )}

        </span>


        ${
            parsed.state

            ?

            `<span class="badge ${stateClass}">
            ${esc(parsed.state)}
            </span>`

            :

            ""
        }


        <span class="score">

        ${
            parsed.score !== null

            ?

            `${parsed.score}/10`

            :

            ""
        }

        ${deltaHTML}

        </span>

    </div>


    <div class="bubble">

    ${formatReaction(
        item.reaction
    )}

    </div>

    `;


    return wrapper;

}


/* ================================= */
/* LOAD FEED */
/* ================================= */

async function loadFeed() {

    const status =

        document.getElementById(
            "status"
        );


    try {

        const response =

            await fetch(

                `/feed/${encodeURIComponent(key)}?limit=40`,

                {
                    cache: "no-store"
                }

            );


        if (!response.ok)

            throw new Error(
                "HTTP " +
                response.status
            );


        const data =

            await response.json();


        const feed =

            document.getElementById(
                "feed"
            );


        const items =

            data.items || [];


        if (firstLoad) {

            feed.innerHTML =
                "";


            if (!items.length) {

                feed.innerHTML = `

                <div class="empty">

                Oczekiwanie na pierwszą analizę...

                </div>

                `;

            }

        }


        items.forEach(item => {

            const id =

                String(

                    item.id ||

                    item.snapshot_id ||

                    item.market_time_uk

                );


            if (
                lastIds.has(id)
            )

                return;


            if (
                feed.querySelector(
                    ".empty"
                )
            ) {

                feed.innerHTML =
                    "";

            }


            lastIds.add(id);


            const message =

                createMessage(
                    item
                );


            /*
            NOWA ANALIZA
            TRAFIA NA GÓRĘ
            */

            feed.prepend(
                message
            );

        });


        status.textContent =

            "LIVE · odświeżono " +

            new Date()
                .toLocaleTimeString(

                    "en-GB",

                    {

                        hour: "2-digit",

                        minute: "2-digit",

                        second: "2-digit"

                    }

                );


        firstLoad =
            false;

    }


    catch(error) {

        console.error(
            error
        );


        status.textContent =

            "Błąd połączenia · ponawiam...";

    }

}


/* ================================= */
/* START */
/* ================================= */

loadAccount();

loadFeed();

setInterval(
    loadFeed,
    5000
);

</script>

</body>

</html>
"""

    return html.replace(
        "__SAFE_KEY__",
        safe_key
    )
