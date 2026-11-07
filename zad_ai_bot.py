# -*- coding: utf-8 -*-
"""
ZAD AI BOT - all in one file
Telegram AI assistant: button-driven menu, multi-provider fallback
(OpenAI -> Gemini -> Cloudflare Workers AI), file debugging/fixing,
on-demand file generation, premium icon UI with quote-style text.

Install:
    pip install -U pyrogram tgcrypto aiohttp

Run (env vars, see .env.example / README):
    python zad_ai_bot.py
"""
import asyncio
import functools
import html
import json
import logging
import os
import time
import traceback
from collections import defaultdict, deque
from typing import List, Optional

import aiohttp
from pyrogram import Client, filters
from pyrogram.enums import ParseMode
from pyrogram.errors import FloodWait
from pyrogram.types import (CallbackQuery, InlineKeyboardButton,
                            InlineKeyboardMarkup, Message)

# ───────────────────────── CONFIG ─────────────────────────
API_ID = int(os.getenv("API_ID", "0"))
API_HASH = os.getenv("API_HASH", "")
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
OWNER_ID = int(os.getenv("OWNER_ID", "0"))
BRAND = "ZAD AI"

# Put your OWN, freshly-generated keys here via environment variables —
# never hardcode real secrets in this file or commit them to git.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID", "")
CF_API_TOKEN = os.getenv("CF_API_TOKEN", "")
CF_MODEL = os.getenv("CF_MODEL", "@cf/meta/llama-3.1-8b-instruct")

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are ZAD AI, a helpful, concise assistant. Reply in the same "
    "language the user writes in (Hindi/Hinglish/English etc).",
)
MAX_HISTORY = 12
FILE_THRESHOLD = 3500
MAX_READ_BYTES = 2_000_000
DATA_FILE = "zad_ai_data.json"

# ───────────────────────── ICONS (full set, every one used) ─────────────────────────
I = {
    "ok": "✓", "no": "\u2718", "warn": "\u26A0", "arrow": "\u2192",
    "bullet": "\u2022", "tri": "\u25B8", "diamond": "\u25C6", "star": "\u2605",
    "spark": "\u2726", "back": "↲", "fwd": "\u25B6", "plus": "\u2295",
    "minus": "\u2296", "rec": "\u25C9", "rec_off": "\u25CB",
    "div": "\u2501" * 16, "div_eq": "\u2550" * 16, "div_dash": "\u2508" * 16,
    "block_on": "\u25A0", "block_off": "\u25A1",
    "border_top": "\u2550" * 16, "border_mid": "\u2501" * 16, "border_bot": "\u2550" * 16,
    "play": "‣", "stop": "\u25A0", "pause": "\u2759\u2759", "refresh": "\u21BB",
    "running": "\u25B6", "stopped": "■", "restarting": "\u21BB", "stop_bot": "■",
    "lock": "\u25A3", "unlock": "\u25A2", "secure": "\u25C8", "key": "\u2756",
    "shield": "\u25C7", "ban": "\u2694", "trash": "\u2716", "eye": "\u25C9",
    "user": "\u25C8", "users": "\u25CE", "crown": "\u2654",
    "wallet": "\u25C6", "premium": "⌬", "lifetime": "\u2736", "gift": "\u2726",
    "ticket": "\u273F", "trophy": "\u2605",
    "graph": "\u25AA", "stats": "\u25AA", "chart_up": "\u25B2", "plan": "\u25A4",
    "broadcast": "⚑", "chat": "\u25AB",
    "folder": "\u25B8", "upload": "\u25B4", "download": "\u25BE", "cloud": "\u2601",
    "settings": "⚙", "cog": "\u2699", "bolt": "\u26A1", "clock": "\u23F1",
    "fix": "\u2692", "wrench": "\u2692", "gen": "\u270E",
}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("zad-ai")

if not (API_ID and API_HASH and BOT_TOKEN):
    raise SystemExit("Set API_ID, API_HASH, BOT_TOKEN env vars first.")
if not (OPENAI_API_KEY or GEMINI_API_KEY or (CF_ACCOUNT_ID and CF_API_TOKEN)):
    log.warning("No AI provider keys set — bot will reply with an error until you add at least one.")

bot = Client("zad_ai_bot", api_id=API_ID, api_hash=API_HASH, bot_token=BOT_TOKEN,
             parse_mode=ParseMode.HTML)

# ───────────────────────── STATE ─────────────────────────
HISTORY: dict = defaultdict(lambda: deque(maxlen=MAX_HISTORY))
USERS: set = set()
CHATS: set = set()
STATS = {"openai": 0, "gemini": 0, "cloudflare": 0, "fail": 0, "fixes": 0, "generated": 0}
MODE: dict = {}          # user_id -> "chat" | "await_gen_desc" | "await_fix_file" | "await_gen_type"
GEN_TYPE: dict = {}       # user_id -> chosen extension while awaiting description
STARTED = time.time()
_http: Optional[aiohttp.ClientSession] = None


def esc(s) -> str:
    return html.escape(str(s or ""))


def quote(text: str) -> str:
    """Telegram HTML blockquote - the 'quote style' text block."""
    return f"<blockquote>{text}</blockquote>"


def load_data():
    try:
        with open(DATA_FILE, encoding="utf-8") as f:
            d = json.load(f)
        USERS.update(d.get("users", []))
        CHATS.update(d.get("chats", []))
    except Exception:
        pass


def save_data():
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump({"users": list(USERS), "chats": list(CHATS)}, f)
    except Exception:
        pass


def safe(fn):
    @functools.wraps(fn)
    async def wrapper(client, update):
        try:
            await fn(client, update)
        except FloodWait as e:
            await asyncio.sleep(e.value + 1)
        except Exception as e:
            log.error("handler error: %s\n%s", e, traceback.format_exc())
            try:
                if isinstance(update, CallbackQuery):
                    await update.answer(f"{I['no']} Error, try again", show_alert=False)
                else:
                    await update.reply_text(
                        f"{I['warn']} <b>Something went wrong.</b> {I['shield']} It's logged - try again, "
                        f"or tap {I['back']} Menu below.",
                        reply_markup=main_menu())
            except Exception:
                pass
    return wrapper


async def http() -> aiohttp.ClientSession:
    global _http
    if _http is None or _http.closed:
        _http = aiohttp.ClientSession()
    return _http


# ───────────────────── AI PROVIDERS (fallback chain) ─────────────────────
async def ask_openai(messages: List[dict]) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("openai not configured")
    s = await http()
    payload = {"model": OPENAI_MODEL, "messages": messages, "temperature": 0.7}
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    async with s.post("https://api.openai.com/v1/chat/completions",
                      json=payload, headers=headers,
                      timeout=aiohttp.ClientTimeout(total=60)) as r:
        data = await r.json()
        if r.status != 200:
            raise RuntimeError(f"openai {r.status}: {data.get('error', {}).get('message', data)}")
        return data["choices"][0]["message"]["content"].strip()


async def ask_gemini(messages: List[dict]) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("gemini not configured")
    s = await http()
    sys_text = next((m["content"] for m in messages if m["role"] == "system"), "")
    contents = []
    for m in messages:
        if m["role"] == "system":
            continue
        role = "model" if m["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})
    payload = {"contents": contents}
    if sys_text:
        payload["systemInstruction"] = {"parts": [{"text": sys_text}]}
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}")
    async with s.post(url, json=payload,
                      timeout=aiohttp.ClientTimeout(total=60)) as r:
        data = await r.json()
        if r.status != 200:
            raise RuntimeError(f"gemini {r.status}: {data.get('error', {}).get('message', data)}")
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()


async def ask_cloudflare(messages: List[dict]) -> str:
    if not (CF_ACCOUNT_ID and CF_API_TOKEN):
        raise RuntimeError("cloudflare not configured")
    s = await http()
    url = (f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}"
           f"/ai/run/{CF_MODEL}")
    headers = {"Authorization": f"Bearer {CF_API_TOKEN}", "Content-Type": "application/json"}
    async with s.post(url, json={"messages": messages}, headers=headers,
                      timeout=aiohttp.ClientTimeout(total=60)) as r:
        data = await r.json()
        if r.status != 200 or not data.get("success", True):
            raise RuntimeError(f"cloudflare {r.status}: {data.get('errors', data)}")
        result = data.get("result", {})
        return (result.get("response") or "").strip()


PROVIDERS = [("openai", ask_openai), ("gemini", ask_gemini), ("cloudflare", ask_cloudflare)]


async def ask_ai(messages: List[dict]) -> tuple:
    errors = []
    for name, fn in PROVIDERS:
        try:
            text = await fn(messages)
            if text:
                STATS[name] += 1
                return text, name
        except Exception as e:
            errors.append(f"{name}: {e}")
            log.warning("%s failed, trying next provider: %s", name, e)
            continue
    STATS["fail"] += 1
    raise RuntimeError(" | ".join(errors) or "no provider configured")


# ───────────────────────── MENUS (button-first UI) ─────────────────────────
def main_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"🟢 {I['chat']} Ask AI", callback_data="m|ask"),
         InlineKeyboardButton(f"🔵 {I['gen']} Generate File", callback_data="m|gen")],
        [InlineKeyboardButton(f"🟠 {I['fix']} Fix My File", callback_data="m|fix"),
         InlineKeyboardButton(f"🟣 {I['stats']} Status", callback_data="m|status")],
        [InlineKeyboardButton(f"🟡 {I['trash']} Clear Memory", callback_data="m|clear"),
         InlineKeyboardButton(f"⚪ {I['settings']} Help", callback_data="m|help")],
        [InlineKeyboardButton(f"{I['crown']} {BRAND} {I['premium']}", callback_data="m|about")],
    ])


def back_menu(extra_row=None) -> InlineKeyboardMarkup:
    rows = [extra_row] if extra_row else []
    rows.append([InlineKeyboardButton(f"{I['back']} Menu", callback_data="m|home")])
    return InlineKeyboardMarkup(rows)


def chat_markup() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"🟡 {I['trash']} Clear", callback_data="m|clear"),
         InlineKeyboardButton(f"🔵 {I['download']} As File", callback_data="m|asfile"),
         InlineKeyboardButton(f"🟢 {I['fwd']} Again", callback_data="m|again")],
        [InlineKeyboardButton(f"{I['back']} Menu", callback_data="m|home")],
    ])


def gen_type_menu() -> InlineKeyboardMarkup:
    types = [("txt", "Text"), ("md", "Markdown"), ("py", "Python"),
             ("json", "JSON"), ("html", "HTML"), ("csv", "CSV")]
    rows = []
    row = []
    for ext, _label in types:
        row.append(InlineKeyboardButton(f"🔹 .{ext}", callback_data=f"gt|{ext}"))
        if len(row) == 3:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([InlineKeyboardButton(f"{I['back']} Menu", callback_data="m|home")])
    return InlineKeyboardMarkup(rows)


async def send_text_or_file(m: Message, title: str, body: str, footer: str,
                            force_file: bool, filename: str = "reply.txt"):
    if force_file or len(body) > FILE_THRESHOLD:
        path = f"/tmp/zad_{m.chat.id}_{int(time.time())}_{filename}"
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(body)
            await m.reply_document(
                path, caption=f"{title}\n{footer}" if title else footer,
                reply_markup=chat_markup())
        finally:
            try:
                os.remove(path)
            except Exception:
                pass
    else:
        text = f"{title}\n{quote(esc(body))}\n{footer}" if title else f"{quote(esc(body))}\n{footer}"
        if len(text) > 4000:
            text = text[:3900] + f"\n{I['warn']} <i>(truncated - tap As File for full text)</i>"
        await m.reply_text(text, reply_markup=chat_markup(), disable_web_page_preview=True)


# ───────────────────────── AI HELPERS ─────────────────────────
async def ai_answer(chat_id: int, question: str) -> tuple:
    hist = HISTORY[chat_id]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(hist)
    messages.append({"role": "user", "content": question})
    answer, provider = await ask_ai(messages)
    hist.append({"role": "user", "content": question})
    hist.append({"role": "assistant", "content": answer})
    save_data()
    return answer, provider


async def handle_question(m: Message, question: str, force_file: bool = False):
    if not question.strip():
        return await m.reply_text(f"{I['warn']} Likho kya poochna hai.", reply_markup=main_menu())
    if m.from_user:
        USERS.add(m.from_user.id)
    CHATS.add(m.chat.id)
    wait = await m.reply_text(f"{I['rec']} {I['play']} <b>Thinking...</b> {I['refresh']}")
    try:
        answer, provider = await ai_answer(m.chat.id, question)
    except Exception as e:
        return await wait.edit_text(
            f"{I['no']} <b>All AI providers failed.</b>\n"
            f"{I['warn']} Check your API keys / quota in Render env vars.\n"
            f"{quote(esc(str(e))[:300])}",
            reply_markup=back_menu())
    await wait.delete()
    footer = f"{I['diamond']} <i>via {provider}</i> {I['secure']} {I['star']}"
    await send_text_or_file(m, "", answer, footer, force_file, "answer.txt")


async def read_text_file(path: str) -> Optional[str]:
    try:
        size = os.path.getsize(path)
        if size > MAX_READ_BYTES:
            return None
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(60_000)
    except Exception:
        return None


FIX_PROMPT = (
    "You are a senior software engineer. The user sent a file that has an "
    "error or problem. Diagnose the issue clearly, then give the corrected "
    "FULL file content. Format your answer EXACTLY as:\n"
    "ISSUES:\n<bulleted list of problems found>\n\nFIXED_FILE:\n```\n<full corrected file content>\n```"
)


async def fix_file(m: Message, path: str, filename: str, user_note: str = ""):
    content = await read_text_file(path)
    if content is None:
        return await m.reply_text(
            f"{I['no']} {I['ban']} Couldn't read this file - it may be binary, "
            f"too large (>{MAX_READ_BYTES // 1_000_000}MB), or not plain text.",
            reply_markup=back_menu())
    wait = await m.reply_text(f"{I['fix']} {I['wrench']} <b>Analyzing</b> <code>{esc(filename)}</code> ...")
    question = (f"File name: {filename}\n"
               f"User note: {user_note or '(none - find and fix any errors or problems)'}\n\n"
               f"---FILE CONTENT---\n{content}")
    try:
        messages = [{"role": "system", "content": FIX_PROMPT},
                   {"role": "user", "content": question}]
        answer, provider = await ask_ai(messages)
    except Exception as e:
        return await wait.edit_text(
            f"{I['no']} <b>Could not analyze file.</b>\n{quote(esc(str(e))[:300])}",
            reply_markup=back_menu())
    await wait.delete()
    STATS["fixes"] += 1

    issues, fixed = answer, ""
    if "FIXED_FILE:" in answer:
        issues, _, rest = answer.partition("FIXED_FILE:")
        issues = issues.replace("ISSUES:", "").strip()
        fixed = rest.strip()
        if "```" in fixed:
            parts = fixed.split("```")
            fixed = parts[1] if len(parts) > 1 else fixed
            first_line, _, remainder = fixed.partition("\n")
            if first_line.strip().isalpha():
                fixed = remainder

    await m.reply_text(
        f"{I['shield']} <b>Diagnosis</b> <i>via {provider}</i>\n{quote(esc(issues)[:3500])}",
        reply_markup=back_menu() if not fixed.strip() else None)

    if fixed.strip():
        base, dot, ext = filename.rpartition(".")
        out_name = f"{base or 'fixed'}_fixed.{ext or 'txt'}"
        path2 = f"/tmp/zad_fix_{m.chat.id}_{int(time.time())}_{out_name}"
        try:
            with open(path2, "w", encoding="utf-8") as f:
                f.write(fixed)
            await m.reply_document(
                path2,
                caption=f"{I['ok']} {I['download']} <b>Fixed file ready</b> {I['trophy']}",
                reply_markup=back_menu())
        finally:
            try:
                os.remove(path2)
            except Exception:
                pass


GEN_LABELS = {"txt": "plain text", "md": "Markdown", "py": "Python code",
             "json": "JSON", "html": "HTML", "csv": "CSV"}


async def generate_file(m: Message, description: str, ext: str):
    label = GEN_LABELS.get(ext, ext)
    wait = await m.reply_text(f"{I['gen']} {I['play']} <b>Generating</b> .{ext} file ...")
    prompt = (f"Generate a complete, correct {label} file for this request:\n{description}\n\n"
             f"Reply with ONLY the raw file content, no explanation, no markdown fences.")
    try:
        messages = [{"role": "system", "content": "You write clean, complete, working files only."},
                   {"role": "user", "content": prompt}]
        content, provider = await ask_ai(messages)
    except Exception as e:
        return await wait.edit_text(
            f"{I['no']} <b>Generation failed.</b>\n{quote(esc(str(e))[:300])}",
            reply_markup=back_menu())
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        if content.rstrip().endswith("```"):
            content = content.rstrip()[:-3]
    await wait.delete()
    STATS["generated"] += 1
    name = f"zad_generated_{int(time.time())}.{ext}"
    path = f"/tmp/{name}"
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        await m.reply_document(
            path,
            caption=(f"{I['ok']} {I['gift']} <b>File generated</b> <i>via {provider}</i> {I['star']}\n"
                    f"{I['diamond']} <code>{esc(name)}</code>"),
            reply_markup=back_menu())
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


# ───────────────────────── TEXT BLOCKS ─────────────────────────
def start_text() -> str:
    return (
        f"{I['border_top']}\n{I['spark']} <b>{BRAND} BOT</b> {I['premium']}\n{I['border_mid']}\n"
        f"{I['rec']} <b>Status:</b> {I['running']} Online\n"
        f"{I['bolt']} Multi-provider AI (auto fallback)\n"
        f"{I['unlock']} OpenAI {I['arrow']} Gemini {I['arrow']} Cloudflare\n"
        f"{I['fix']} Fixes broken files automatically\n"
        f"{I['gen']} Generates any file you describe\n"
        f"{I['chat']} Remembers your last {MAX_HISTORY} messages\n"
        f"{I['div_eq']}\n"
        f"{I['wallet']} <b>Plan:</b> {I['lifetime']} Free {I['chart_up']} forever\n"
        f"{I['trophy']} <b>Access:</b> {I['block_on']} All features unlocked\n"
        f"{I['border_bot']}\n\n"
        f"{I['arrow']} Use the buttons below - no commands needed."
    )


def help_text() -> str:
    return (
        f"{I['border_top']}\n{I['spark']} <b>{BRAND} HELP</b>\n{I['border_mid']}\n"
        f"{I['tri']} <b>{I['chat']} Ask AI</b> - type anything, get an AI answer\n"
        f"{I['tri']} <b>{I['gen']} Generate File</b> - pick a type, describe it, get a file\n"
        f"{I['tri']} <b>{I['fix']} Fix My File</b> - send a broken file, get a diagnosis + fixed file\n"
        f"{I['tri']} <b>{I['stats']} Status</b> - which AI providers are working\n"
        f"{I['tri']} <b>{I['trash']} Clear Memory</b> - forget this chat's history\n"
        f"{I['div_dash']}\n"
        f"{I['key']} Private per chat  {I['ban']} No spam\n"
        f"{I['graph']} Commands also work: /ask /file /clear /status\n"
        f"{I['div_dash']}\n"
        f"{I['bullet']} {I['plan']} <b>Plan:</b> {I['ticket']} Free, unlimited, no card needed\n"
        f"{I['border_bot']}"
    )


def provider_block(name: str, configured: bool, used: int) -> str:
    dot = I['block_on'] if configured else I['block_off']
    lock = I['unlock'] if configured else I['lock']
    trend = I['plus'] if used > 0 else I['minus']
    bar = I['chart_up'] * min(5, 1 + used // 5) if used else I['stopped']
    return f"{dot} {lock} <b>{name}</b> - {I['graph']} used {used}x {trend}  {bar}"


def status_text() -> str:
    return (
        f"{I['border_top']}\n{I['stats']} <b>Provider status</b>\n{I['border_bot']}\n"
        f"{provider_block('OpenAI', bool(OPENAI_API_KEY), STATS['openai'])}\n"
        f"{provider_block('Gemini', bool(GEMINI_API_KEY), STATS['gemini'])}\n"
        f"{provider_block('Cloudflare', bool(CF_ACCOUNT_ID and CF_API_TOKEN), STATS['cloudflare'])}\n"
        f"{I['div_dash']}\n"
        f"{I['fix']} Files fixed: {STATS['fixes']}   {I['gen']} Files generated: {STATS['generated']}\n"
        f"{I['ban'] if STATS['fail'] else I['shield']} Failed (all down): {STATS['fail']}x"
    )


# ───────────────────────── COMMANDS ─────────────────────────
@bot.on_message(filters.command(["start"]))
@safe
async def cmd_start(c, m: Message):
    if m.from_user:
        USERS.add(m.from_user.id)
    CHATS.add(m.chat.id)
    MODE.pop(m.from_user.id if m.from_user else 0, None)
    save_data()
    await m.reply_text(start_text(), reply_markup=main_menu())


@bot.on_message(filters.command(["help"]))
@safe
async def cmd_help(c, m):
    await m.reply_text(help_text(), reply_markup=back_menu())


@bot.on_message(filters.command(["clear"]))
@safe
async def cmd_clear(c, m):
    HISTORY.pop(m.chat.id, None)
    await m.reply_text(f"{I['trash']} {I['stop_bot']} <b>Memory cleared</b> for this chat.",
                       reply_markup=back_menu())


@bot.on_message(filters.command(["ping"]))
@safe
async def cmd_ping(c, m):
    t0 = time.time()
    r = await m.reply_text(f"{I['rec_off']} <b>Pinging...</b>")
    ms = (time.time() - t0) * 1000
    up = time.time() - STARTED
    h, rem = divmod(int(up), 3600)
    mins, sec = divmod(rem, 60)
    state = I['play'] if ms < 300 else (I['pause'] if ms < 800 else I['stop'])
    await r.edit_text(f"{I['bolt']} <b>Pong:</b> <code>{ms:.0f} ms</code> {state}\n"
                      f"{I['clock']} <b>Uptime:</b> <code>{h}:{mins:02d}:{sec:02d}</code>\n"
                      f"{I['div']}\n{I['settings']} {I['cog']} All systems nominal",
                      reply_markup=back_menu())


@bot.on_message(filters.command(["status"]))
@safe
async def cmd_status(c, m):
    await m.reply_text(status_text(), reply_markup=back_menu())


@bot.on_message(filters.command(["stats"]))
@safe
async def cmd_owner_stats(c, m):
    if not m.from_user or m.from_user.id != OWNER_ID:
        return
    await m.reply_text(f"{I['stats']} {I['chart_up']} <b>Owner stats</b>\n{I['div_dash']}\n"
                       f"{I['users']} Users: <b>{len(USERS)}</b>   {I['chat']} Chats: <b>{len(CHATS)}</b>\n"
                       f"{I['graph']} OpenAI: {STATS['openai']} | Gemini: {STATS['gemini']} | "
                       f"Cloudflare: {STATS['cloudflare']} | {I['ban']} Failed: {STATS['fail']}\n"
                       f"{I['fix']} Fixed: {STATS['fixes']}  {I['gen']} Generated: {STATS['generated']}\n"
                       f"{I['wallet']} Plan: {I['lifetime']} Free for everyone")


@bot.on_message(filters.command(["broadcast"]))
@safe
async def cmd_broadcast(c, m: Message):
    if not m.from_user or m.from_user.id != OWNER_ID:
        return
    text = " ".join(m.command[1:]).strip()
    src = m.reply_to_message
    if not text and not src:
        return await m.reply_text(f"{I['warn']} Usage: <code>/broadcast text</code> or reply to a message.")
    ok = bad = 0
    for cid in list(CHATS | USERS):
        try:
            if src:
                await src.copy(cid)
            else:
                await bot.send_message(cid, text)
            ok += 1
        except FloodWait as e:
            await asyncio.sleep(e.value + 1)
        except Exception:
            bad += 1
        await asyncio.sleep(0.3)
    await m.reply_text(f"{I['broadcast']} {I['gift']} <b>Broadcast done</b>\n{I['ok']} Sent: {ok}\n{I['no']} Failed: {bad}")


@bot.on_message(filters.command(["ask"]))
@safe
async def cmd_ask(c, m: Message):
    await handle_question(m, " ".join(m.command[1:]))


@bot.on_message(filters.command(["file"]))
@safe
async def cmd_file(c, m: Message):
    await handle_question(m, " ".join(m.command[1:]), force_file=True)


# ───────────────────────── TEXT / FILE ROUTING ─────────────────────────
@bot.on_message(filters.text & filters.private & ~filters.command([
    "start", "help", "clear", "ping", "status", "stats", "broadcast", "ask", "file"]))
@safe
async def on_text(c, m: Message):
    uid = m.from_user.id if m.from_user else 0
    mode = MODE.get(uid, "chat")

    if mode == "await_gen_desc":
        ext = GEN_TYPE.get(uid, "txt")
        MODE[uid] = "chat"
        return await generate_file(m, m.text, ext)

    if mode == "await_fix_file":
        return await m.reply_text(
            f"{I['warn']} {I['upload']} Text nahi, <b>file attach</b> karke bhejo jisme problem hai.",
            reply_markup=back_menu())

    await handle_question(m, m.text)


@bot.on_message((filters.document | filters.photo) & filters.private)
@safe
async def on_file(c, m: Message):
    uid = m.from_user.id if m.from_user else 0
    mode = MODE.get(uid, "chat")

    if not m.document:
        return await m.reply_text(
            f"{I['warn']} Photos ko samajhna abhi support nahi hai - text/code files bhejo.",
            reply_markup=back_menu())

    if m.document.file_size and m.document.file_size > MAX_READ_BYTES:
        return await m.reply_text(
            f"{I['no']} {I['ban']} File too big (max {MAX_READ_BYTES // 1_000_000}MB).",
            reply_markup=back_menu())

    notice = await m.reply_text(f"{I['cloud']} {I['upload']} <b>Reading file</b> {I['user']} from you...")
    try:
        path = await m.download(file_name="/tmp/")
    except Exception as e:
        return await notice.edit_text(f"{I['no']} Couldn't download file: {esc(str(e))[:200]}",
                                      reply_markup=back_menu())
    try:
        await notice.delete()
    except Exception:
        pass

    filename = m.document.file_name or "file.txt"

    if mode == "await_fix_file":
        MODE[uid] = "chat"
        try:
            await fix_file(m, path, filename, user_note=m.caption or "")
        finally:
            try:
                os.remove(path)
            except Exception:
                pass
        return

    content = await read_text_file(path)
    try:
        os.remove(path)
    except Exception:
        pass
    if content is None:
        return await m.reply_text(
            f"{I['warn']} Could only read plain text/code files right now. Agar isme error fix "
            f"karwana hai to {I['fix']} <b>Fix My File</b> button use karo.",
            reply_markup=back_menu())
    q = (m.caption or "Summarize this file and point out any obvious issues.").strip()
    await handle_question(m, f"{q}\n\n---FILE CONTENT ({esc(filename)})---\n{content}")


# ───────────────────────── CALLBACKS ─────────────────────────
@bot.on_callback_query(filters.regex(r"^m\|"))
@safe
async def cb_menu(c, q: CallbackQuery):
    act = q.data.split("|", 1)[1]
    uid = q.from_user.id

    if act == "home":
        MODE.pop(uid, None)
        await q.answer()
        return await q.message.edit_text(start_text(), reply_markup=main_menu())

    if act == "about":
        return await q.answer(f"{BRAND} - Multi-provider AI, file fixing & generation - free forever.",
                              show_alert=True)

    if act == "help":
        await q.answer()
        return await q.message.edit_text(help_text(), reply_markup=back_menu())

    if act == "status":
        await q.answer()
        return await q.message.edit_text(status_text(), reply_markup=back_menu())

    if act == "clear":
        HISTORY.pop(q.message.chat.id, None)
        MODE.pop(uid, None)
        await q.answer(f"{I['ok']} Memory cleared", show_alert=True)
        return await q.message.edit_text(
            f"{I['trash']} {I['stop_bot']} <b>Memory cleared.</b>", reply_markup=back_menu())

    if act == "ask":
        MODE[uid] = "chat"
        await q.answer()
        return await q.message.edit_text(
            f"{I['chat']} {I['spark']} <b>Ask AI</b>\n"
            f"{quote('Type your question below - koi bhi topic, main reply karunga.')}",
            reply_markup=back_menu())

    if act == "gen":
        MODE[uid] = "await_gen_type"
        await q.answer()
        return await q.message.edit_text(
            f"{I['gen']} {I['folder']} <b>Generate File</b>\n"
            f"{quote('File type choose karo:')}",
            reply_markup=gen_type_menu())

    if act == "fix":
        MODE[uid] = "await_fix_file"
        await q.answer()
        return await q.message.edit_text(
            f"{I['fix']} {I['wrench']} <b>Fix My File</b>\n"
            f"{quote('Apni file bhejo (code/text). Caption me bata sakte ho kya problem hai - nahi bataya to main khud dhoondh lunga.')}",
            reply_markup=back_menu())

    if act == "again":
        src_msg = q.message.reply_to_message
        if not src_msg or not src_msg.text:
            return await q.answer(f"{I['warn']} Original question nahi mili.", show_alert=True)
        await q.answer(f"{I['fwd']} Regenerating...")
        return await handle_question(src_msg, src_msg.text)

    if act == "asfile":
        body = q.message.text or q.message.caption or ""
        path = f"/tmp/zad_asfile_{q.message.chat.id}_{int(time.time())}.txt"
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(body)
            await q.message.reply_document(path, caption=f"{I['download']} {I['eye']} Exported")
        finally:
            try:
                os.remove(path)
            except Exception:
                pass
        return await q.answer(f"{I['ok']} Sent as file")


@bot.on_callback_query(filters.regex(r"^gt\|"))
@safe
async def cb_gen_type(c, q: CallbackQuery):
    ext = q.data.split("|", 1)[1]
    uid = q.from_user.id
    GEN_TYPE[uid] = ext
    MODE[uid] = "await_gen_desc"
    await q.answer(f"{I['ok']} .{ext} selected")
    await q.message.edit_text(
        f"{I['gen']} {I['tri']} <b>.{ext}</b> file\n"
        f"{quote('Ab describe karo file me kya chahiye - jitna detail doge utni accurate file banegi.')}",
        reply_markup=back_menu())


# ───────────────────────── MAIN ─────────────────────────
async def main():
    load_data()
    await bot.start()
    me = await bot.get_me()
    log.info("%s ready as @%s", BRAND, me.username)
    if OWNER_ID:
        try:
            await bot.send_message(OWNER_ID, f"{I['running']} <b>{BRAND}</b> is online {I['ok']}")
        except Exception:
            pass
    from pyrogram import idle
    await idle()
    await bot.stop()


if __name__ == "__main__":
    while True:
        try:
            asyncio.get_event_loop().run_until_complete(main())
            break
        except KeyboardInterrupt:
            break
        except Exception as e:
            log.error("%s fatal, restarting in 5s: %s\n%s", I['restarting'], e, traceback.format_exc())
            time.sleep(5)
