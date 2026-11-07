# -*- coding: utf-8 -*-
"""
ZAD AI BOT
Telegram Bot API edition
BOT_TOKEN only - no API_ID / API_HASH
"""

import os
import re
import json
import time
import html
import asyncio
import logging
import tempfile
import traceback
from pathlib import Path
from collections import defaultdict, deque

import aiohttp

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.error import TelegramError, RetryAfter, TimedOut, NetworkError
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    Defaults,
    filters,
)

# ================= CONFIG =================

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
OWNER_ID = int(os.getenv("OWNER_ID", "0") or "0")

BRAND = "ZAD AI"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID", "").strip()
CF_API_TOKEN = os.getenv("CF_API_TOKEN", "").strip()
CF_MODEL = os.getenv(
    "CF_MODEL",
    "@cf/meta/llama-3.1-8b-instruct"
)

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are ZAD AI, a helpful, concise assistant. "
    "Reply in the same language the user writes in."
)

MAX_HISTORY = 12
FILE_THRESHOLD = 3500
MAX_READ_BYTES = 2_000_000
DATA_FILE = os.getenv("DATA_FILE", "zad_ai_data.json")

if not BOT_TOKEN:
    raise SystemExit("BOT_TOKEN missing. Add it in Render Environment.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("zad-ai")

# ================= ORIGINAL ICONS =================

I = {
    "ok": "✓", "no": "✘", "warn": "⚠", "arrow": "→",
    "bullet": "•", "tri": "▸", "diamond": "◆", "star": "★",
    "spark": "✦", "back": "↲", "fwd": "▶", "plus": "⊕",
    "minus": "⊖", "rec": "◉", "rec_off": "○",
    "div": "━" * 16, "div_eq": "═" * 16,
    "div_dash": "┈" * 16,
    "block_on": "■", "block_off": "□",
    "border_top": "═" * 16, "border_mid": "━" * 16,
    "border_bot": "═" * 16,
    "play": "‣", "stop": "■", "pause": "❙❙",
    "refresh": "↻", "running": "▶", "stopped": "■",
    "restarting": "↻", "stop_bot": "■",
    "lock": "▣", "unlock": "▢", "secure": "◈",
    "key": "❖", "shield": "◇", "ban": "⚔",
    "trash": "✖", "eye": "◉", "user": "◈",
    "users": "◎", "crown": "♔", "wallet": "◆",
    "premium": "⌬", "lifetime": "✶", "gift": "✦",
    "ticket": "✿", "trophy": "★", "graph": "▪",
    "stats": "▪", "chart_up": "▲", "plan": "▤",
    "broadcast": "⚑", "chat": "▫", "folder": "▸",
    "upload": "▴", "download": "▾", "cloud": "☁",
    "settings": "⚙", "cog": "⚙", "bolt": "⚡",
    "clock": "⏱", "fix": "⚒", "wrench": "⚒",
    "gen": "✎",
}

# ================= STATE =================

HISTORY = defaultdict(lambda: deque(maxlen=MAX_HISTORY))
USERS = set()
CHATS = set()

STATS = {
    "openai": 0,
    "gemini": 0,
    "cloudflare": 0,
    "fail": 0,
    "fixes": 0,
    "generated": 0,
}

MODE = {}
GEN_TYPE = {}
LAST_QUESTION = {}

STARTED = time.time()
_http = None


# ================= HELPERS =================

def esc(value):
    return html.escape(str(value or ""))


def quote(text):
    return f"<blockquote>{text}</blockquote>"


def save_data():
    try:
        data = {
            "users": list(USERS),
            "chats": list(CHATS),
            "history": {
                str(key): list(value)
                for key, value in HISTORY.items()
            },
            "stats": STATS,
        }

        temp = DATA_FILE + ".tmp"

        with open(temp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        os.replace(temp, DATA_FILE)

    except Exception:
        log.exception("Data save failed")


def load_data():
    try:
        with open(DATA_FILE, encoding="utf-8") as f:
            data = json.load(f)

        USERS.update(int(x) for x in data.get("users", []))
        CHATS.update(int(x) for x in data.get("chats", []))

        for key, messages in data.get("history", {}).items():
            HISTORY[int(key)].extend(messages[-MAX_HISTORY:])

        for key in STATS:
            STATS[key] = int(data.get("stats", {}).get(key, 0))

    except FileNotFoundError:
        pass
    except Exception:
        log.exception("Data loading failed")


async def http():
    global _http

    if _http is None or _http.closed:
        _http = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=65)
        )

    return _http


# ================= OPENAI =================

async def ask_openai(messages):
    if not OPENAI_API_KEY:
        raise RuntimeError("OpenAI not configured")

    session = await http()

    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": 0.7,
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    async with session.post(
        "https://api.openai.com/v1/chat/completions",
        json=payload,
        headers=headers,
    ) as response:

        data = await response.json(content_type=None)

        if response.status != 200:
            raise RuntimeError(
                f"OpenAI {response.status}: {data}"
            )

        choices = data.get("choices", [])

        if not choices:
            raise RuntimeError("OpenAI returned no response")

        answer = choices[0]["message"].get("content", "")

        if not answer.strip():
            raise RuntimeError("OpenAI returned empty response")

        return answer.strip()


# ================= GEMINI =================

async def ask_gemini(messages):
    if not GEMINI_API_KEY:
        raise RuntimeError("Gemini not configured")

    session = await http()

    system_text = next(
        (
            item["content"]
            for item in messages
            if item["role"] == "system"
        ),
        ""
    )

    contents = []

    for item in messages:
        if item["role"] == "system":
            continue

        role = (
            "model"
            if item["role"] == "assistant"
            else "user"
        )

        contents.append({
            "role": role,
            "parts": [{"text": item["content"]}]
        })

    payload = {"contents": contents}

    if system_text:
        payload["systemInstruction"] = {
            "parts": [{"text": system_text}]
        }

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent"
        f"?key={GEMINI_API_KEY}"
    )

    async with session.post(url, json=payload) as response:
        data = await response.json(content_type=None)

        if response.status != 200:
            raise RuntimeError(
                f"Gemini {response.status}: {data}"
            )

        candidates = data.get("candidates", [])

        if not candidates:
            raise RuntimeError("Gemini returned no candidates")

        parts = candidates[0].get("content", {}).get("parts", [])

        answer = "\n".join(
            part.get("text", "")
            for part in parts
            if part.get("text")
        )

        if not answer.strip():
            raise RuntimeError("Gemini returned empty response")

        return answer.strip()


# ================= CLOUDFLARE =================

async def ask_cloudflare(messages):
    if not (CF_ACCOUNT_ID and CF_API_TOKEN):
        raise RuntimeError("Cloudflare not configured")

    session = await http()

    url = (
        "https://api.cloudflare.com/client/v4/accounts/"
        f"{CF_ACCOUNT_ID}/ai/run/{CF_MODEL}"
    )

    headers = {
        "Authorization": f"Bearer {CF_API_TOKEN}",
        "Content-Type": "application/json",
    }

    async with session.post(
        url,
        json={"messages": messages},
        headers=headers,
    ) as response:

        data = await response.json(content_type=None)

        if response.status != 200 or not data.get("success", True):
            raise RuntimeError(f"Cloudflare error: {data}")

        answer = data.get("result", {}).get("response", "")

        if not answer.strip():
            raise RuntimeError("Cloudflare returned empty response")

        return answer.strip()


# ================= PROVIDER FALLBACK =================

PROVIDERS = [
    ("openai", ask_openai),
    ("gemini", ask_gemini),
    ("cloudflare", ask_cloudflare),
]


async def ask_ai(messages):
    errors = []

    for name, provider in PROVIDERS:
        try:
            answer = await provider(messages)

            if answer:
                STATS[name] += 1
                save_data()
                return answer, name

        except Exception as error:
            errors.append(f"{name}: {error}")
            log.warning("%s failed: %s", name, error)

    STATS["fail"] += 1
    save_data()

    raise RuntimeError(" | ".join(errors) or "No AI provider configured")


# ================= MENUS =================

def main_menu():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(
                f"🟢 {I['chat']} Ask AI",
                callback_data="m|ask"
            ),
            InlineKeyboardButton(
                f"🔵 {I['gen']} Generate File",
                callback_data="m|gen"
            ),
        ],
        [
            InlineKeyboardButton(
                f"🟠 {I['fix']} Fix My File",
                callback_data="m|fix"
            ),
            InlineKeyboardButton(
                f"🟣 {I['stats']} Status",
                callback_data="m|status"
            ),
        ],
        [
            InlineKeyboardButton(
                f"🟡 {I['trash']} Clear Memory",
                callback_data="m|clear"
            ),
            InlineKeyboardButton(
                f"⚪ {I['settings']} Help",
                callback_data="m|help"
            ),
        ],
        [
            InlineKeyboardButton(
                f"{I['crown']} {BRAND} {I['premium']}",
                callback_data="m|about"
            ),
        ],
    ])


def back_menu(extra_row=None):
    rows = [extra_row] if extra_row else []

    rows.append([
        InlineKeyboardButton(
            f"{I['back']} Menu",
            callback_data="m|home"
        )
    ])

    return InlineKeyboardMarkup(rows)


def chat_markup():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(
                f"🟡 {I['trash']} Clear",
                callback_data="m|clear"
            ),
            InlineKeyboardButton(
                f"🔵 {I['download']} As File",
                callback_data="m|asfile"
            ),
            InlineKeyboardButton(
                f"🟢 {I['fwd']} Again",
                callback_data="m|again"
            ),
        ],
        [
            InlineKeyboardButton(
                f"{I['back']} Menu",
                callback_data="m|home"
            )
        ],
    ])


def gen_type_menu():
    types = ["txt", "md", "py", "json", "html", "csv"]
    rows = []

    for index in range(0, len(types), 3):
        rows.append([
            InlineKeyboardButton(
                f"🔹 .{ext}",
                callback_data=f"gt|{ext}"
            )
            for ext in types[index:index + 3]
        ])

    rows.append([
        InlineKeyboardButton(
            f"{I['back']} Menu",
            callback_data="m|home"
        )
    ])

    return InlineKeyboardMarkup(rows)
# =========================
# PART 2/3 — BOT FUNCTIONS
# =========================

import os
import re
import json
import html
import tempfile
from pathlib import Path


def start_text(user):
    name = html.escape(user.first_name or "User")
    return (
        f"👋 <b>Welcome, {name}!</b>\n\n"
        "🤖 <b>ZAD AI Assistant</b>\n"
        "Your AI assistant is ready.\n\n"
        "Choose an option from the menu below."
    )


def help_text():
    return (
        "📚 <b>ZAD AI — Help</b>\n\n"
        "💬 <b>AI Chat</b>\n"
        "Send a message and get an AI response.\n\n"
        "🧠 <b>AI Providers</b>\n"
        "OpenAI and Gemini are supported, with automatic fallback.\n\n"
        "📁 <b>File Tools</b>\n"
        "Send a supported text or code file for analysis or fixing.\n\n"
        "⚙️ <b>Commands</b>\n"
        "/start — Open main menu\n"
        "/help — Show help\n"
        "/clear — Clear chat history\n"
        "/ping — Check bot response\n"
        "/status — Provider status\n"
        "/ask your question — Ask AI"
    )


def status_text():
    openai_state = "Configured" if OPENAI_API_KEY else "Not configured"
    gemini_state = "Configured" if GEMINI_API_KEY else "Not configured"
    cloudflare_state = (
        "Configured" if CLOUDFLARE_API_TOKEN else "Not configured"
    )

    return (
        "⚙️ <b>ZAD AI — Status</b>\n\n"
        f"🟢 Telegram: Connected\n"
        f"🤖 OpenAI: {openai_state}\n"
        f"✨ Gemini: {gemini_state}\n"
        f"☁️ Cloudflare: {cloudflare_state}\n\n"
        "Automatic provider fallback is enabled."
    )


def safe_html(text):
    """Escape AI output so Telegram HTML parsing does not fail."""
    text = str(text or "")
    parts = re.split(r"```(?:[a-zA-Z0-9_+-]+)?\n?", text)

    output = []
    for index, part in enumerate(parts):
        escaped = html.escape(part.strip("\n"))
        if index % 2:
            output.append(f"<pre>{escaped}</pre>")
        else:
            output.append(escaped)

    return "\n".join(output)


async def send_long_message(message, text, limit=3900):
    """Send long text in Telegram-safe chunks."""
    text = str(text or "")
    if not text:
        text = "No response was returned."

    chunks = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(text[:cut])
        text = text[cut:].lstrip()

    if text:
        chunks.append(text)

    for chunk in chunks:
        try:
            await message.reply_text(
                safe_html(chunk),
                parse_mode="HTML",
                disable_web_page_preview=True
            )
        except Exception:
            await message.reply_text(chunk[:4000])


def get_user_history(user_id):
    key = str(user_id)
    if key not in HISTORY:
        HISTORY[key] = []
    return HISTORY[key]


async def handle_question(message, user, question, context):
    question = (question or "").strip()

    if not question:
        await message.reply_text("Please send a question.")
        return

    if len(question) > 12000:
        await message.reply_text(
            "Your message is too long. Please send a shorter question."
        )
        return

    user_id = str(user.id)
    history = get_user_history(user_id)

    history.append({
        "role": "user",
        "content": question
    })

    # Keep only recent conversation messages.
    if len(history) > MAX_HISTORY:
        del history[:-MAX_HISTORY]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)

    wait_message = await message.reply_text("⏳ Thinking...")

    try:
        answer, provider = await ask_ai(messages)

        if not answer:
            history.pop()
            await wait_message.edit_text(
                "⚠️ No AI provider returned a response. Please try again."
            )
            return

        history.append({
            "role": "assistant",
            "content": answer
        })

        if len(history) > MAX_HISTORY:
            del history[:-MAX_HISTORY]

        STATS["requests"] = STATS.get("requests", 0) + 1
        save_data()

        provider_label = html.escape(str(provider or "AI"))
        response = (
            f"🤖 <b>{provider_label}</b>\n\n"
            f"{safe_html(answer)}"
        )

        # Send the response in smaller parts.
        await wait_message.delete()
        await send_long_message(message, response)

    except Exception as exc:
        if history and history[-1].get("role") == "user":
            history.pop()

        try:
            await wait_message.edit_text(
                "⚠️ Something went wrong while contacting the AI provider."
            )
        except Exception:
            await message.reply_text(
                "⚠️ Something went wrong while contacting the AI provider."
            )

        print("AI QUESTION ERROR:", repr(exc))


def extract_code(text):
    """Extract the first fenced code block, if present."""
    match = re.search(
        r"```(?:[a-zA-Z0-9_+-]+)?\s*\n?(.*?)```",
        text or "",
        re.DOTALL
    )
    return match.group(1).strip() if match else (text or "").strip()


def clean_filename(name):
    name = os.path.basename(name or "zad_output.txt")
    name = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    return name[:100] or "zad_output.txt"


async def send_text_as_file(message, filename, content, caption=None):
    filename = clean_filename(filename)

    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, filename)

        with open(path, "w", encoding="utf-8") as file:
            file.write(content)

        with open(path, "rb") as file:
            await message.reply_document(
                document=file,
                filename=filename,
                caption=caption or "📄 Your file is ready."
            )


async def generate_file(message, user, context):
    user_id = str(user.id)
    generation_type = GEN_TYPE.get(user_id, "python")

    prompt = LAST_QUESTION.get(user_id, "").strip()

    if not prompt:
        await message.reply_text(
            "Please describe the file you want to generate."
        )
        return

    wait_message = await message.reply_text(
        "🛠️ Generating your file..."
    )

    try:
        request = (
            "Create a complete, working file based on this request.\n"
            f"File type: {generation_type}\n"
            f"User request: {prompt}\n\n"
            "Return the complete file contents in a code block. "
            "Do not omit required sections."
        )

        answer, provider = await ask_ai([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": request}
        ])

        if not answer:
            await wait_message.edit_text(
                "⚠️ File generation failed. Please try again."
            )
            return

        content = extract_code(answer)

        extensions = {
            "python": ".py",
            "py": ".py",
            "html": ".html",
            "css": ".css",
            "javascript": ".js",
            "js": ".js",
            "json": ".json",
            "text": ".txt",
            "txt": ".txt"
        }

        extension = extensions.get(
            str(generation_type).lower(),
            ".txt"
        )

        filename = "zad_generated" + extension

        await wait_message.delete()
        await send_text_as_file(
            message,
            filename,
            content,
            caption=f"✅ Generated using {provider or 'AI'}"
        )

        STATS["generated"] = STATS.get("generated", 0) + 1
        save_data()

    except Exception as exc:
        print("FILE GENERATION ERROR:", repr(exc))
        try:
            await wait_message.edit_text(
                "⚠️ Could not generate the file."
            )
        except Exception:
            pass


async def download_telegram_file(update, context, file_id, filename):
    """Download a Telegram document to a temporary local path."""
    telegram_file = await context.bot.get_file(file_id)

    temp_dir = tempfile.mkdtemp(prefix="zad_ai_")
    filename = clean_filename(filename)
    path = os.path.join(temp_dir, filename)

    await telegram_file.download_to_drive(custom_path=path)
    return path, temp_dir


async def read_text_file(path):
    """Read supported text/code files with a size limit."""
    size = os.path.getsize(path)

    if size > MAX_READ_BYTES:
        raise ValueError(
            "File is too large to process. Maximum supported size is 2 MB."
        )

    with open(path, "rb") as file:
        raw = file.read()

    # Try common text encodings.
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue

    raise ValueError("This file could not be decoded as text.")


async def analyze_uploaded_file(message, user, context, file_text, filename):
    prompt = (
        "Analyze the following uploaded file.\n"
        f"Filename: {filename}\n\n"
        "Explain its purpose, identify possible errors, and suggest "
        "improvements. Do not claim that you executed or tested it.\n\n"
        "FILE CONTENT:\n"
        f"{file_text[:MAX_READ_BYTES]}"
    )

    await handle_question(message, user, prompt, context)


async def fix_uploaded_file(message, user, context, file_text, filename):
    wait_message = await message.reply_text(
        "🔧 Analyzing and fixing your file..."
    )

    try:
        prompt = (
            "Review and fix the following text/code file.\n"
            f"Original filename: {filename}\n\n"
            "Return the complete corrected file in one code block. "
            "Do not intentionally remove existing functionality. "
            "After the code block, briefly list the changes.\n\n"
            "FILE CONTENT:\n"
            f"{file_text[:MAX_READ_BYTES]}"
        )

        answer, provider = await ask_ai([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ])

        if not answer:
            await wait_message.edit_text(
                "⚠️ The AI could not fix this file."
            )
            return

        fixed_content = extract_code(answer)

        original = Path(filename)
        output_name = original.stem + "_fixed" + original.suffix

        await wait_message.delete()

        await send_text_as_file(
            message,
            output_name,
            fixed_content,
            caption=f"✅ Fixed file — {provider or 'AI'}"
        )

        STATS["fixed"] = STATS.get("fixed", 0) + 1
        save_data()

        # Send a short explanation separately.
        explanation = answer.replace(
            f"```{''}", ""
        )
        if len(explanation) > 1500:
            explanation = explanation[-1500:]

        await message.reply_text(
            "📝 <b>Fix summary</b>\n" + safe_html(explanation),
            parse_mode="HTML"
        )

    except Exception as exc:
        print("FILE FIX ERROR:", repr(exc))
        try:
            await wait_message.edit_text(
                "⚠️ File fixing failed."
            )
        except Exception:
            pass


async def process_uploaded_file(update, context, file_id, filename):
    message = update.effective_message
    user = update.effective_user

    path = None
    temp_dir = None

    try:
        path, temp_dir = await download_telegram_file(
            update,
            context,
            file_id,
            filename
        )

        file_text = await read_text_file(path)

        if not file_text.strip():
            await message.reply_text("The uploaded file is empty.")
            return

        user_id = str(user.id)

        # The file action is selected in Part 3.
        action = FILE_ACTION.get(user_id, "analyze")

        if action == "fix":
            await fix_uploaded_file(
                message,
                user,
                context,
                file_text,
                filename
            )
        else:
            await analyze_uploaded_file(
                message,
                user,
                context,
                file_text,
                filename
            )

    except ValueError as exc:
        await message.reply_text(f"⚠️ {html.escape(str(exc))}")

    except Exception as exc:
        print("UPLOAD PROCESS ERROR:", repr(exc))
        await message.reply_text(
            "⚠️ Could not process this file. Please try another text/code file."
        )

    finally:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

        if temp_dir and os.path.isdir(temp_dir):
            try:
                os.rmdir(temp_dir)
            except OSError:
                pass


# End of Part 2/3
# =========================
# PART 3/3 — HANDLERS & STARTUP
# =========================

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    Defaults,
    filters,
)

# User file preferences
if "FILE_ACTION" not in globals():
    FILE_ACTION = {}


def is_owner(user_id):
    return OWNER_ID and str(user_id) == str(OWNER_ID)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not user:
        return

    USERS[str(user.id)] = {
        "id": user.id,
        "name": user.first_name or "User",
        "username": user.username or ""
    }
    save_data()

    await update.effective_message.reply_text(
        start_text(user),
        reply_markup=main_menu(),
        parse_mode=ParseMode.HTML
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        help_text(),
        parse_mode=ParseMode.HTML,
        reply_markup=back_menu()
    )


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    HISTORY[user_id] = []
    save_data()

    await update.effective_message.reply_text(
        "🧹 Chat history cleared."
    )


async def ping_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "🏓 <b>Pong!</b> ZAD AI is responding.",
        parse_mode=ParseMode.HTML
    )


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        status_text(),
        parse_mode=ParseMode.HTML,
        reply_markup=back_menu()
    )


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_owner(update.effective_user.id):
        await update.effective_message.reply_text("⛔ Owner only command.")
        return

    text = (
        "📊 <b>ZAD AI Statistics</b>\n\n"
        f"👥 Users: {len(USERS)}\n"
        f"💬 Requests: {STATS.get('requests', 0)}\n"
        f"📄 Generated: {STATS.get('generated', 0)}\n"
        f"🔧 Fixed: {STATS.get('fixed', 0)}"
    )

    await update.effective_message.reply_text(
        text,
        parse_mode=ParseMode.HTML
    )


async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = " ".join(context.args).strip()

    if not question:
        await update.effective_message.reply_text(
            "Usage: /ask your question"
        )
        return

    await handle_question(
        update.effective_message,
        update.effective_user,
        question,
        context
    )


async def broadcast_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not is_owner(update.effective_user.id):
        await update.effective_message.reply_text("⛔ Owner only command.")
        return

    message_text = " ".join(context.args).strip()

    if not message_text:
        await update.effective_message.reply_text(
            "Usage: /broadcast your message"
        )
        return

    success = 0
    failed = 0

    await update.effective_message.reply_text(
        "📢 Starting broadcast..."
    )

    for user_id in list(USERS.keys()):
        try:
            await context.bot.send_message(
                chat_id=int(user_id),
                text=message_text
            )
            success += 1
        except Exception:
            failed += 1

    await update.effective_message.reply_text(
        f"✅ Sent: {success}\n❌ Failed: {failed}"
    )


async def file_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    FILE_ACTION[user_id] = "analyze"

    await update.effective_message.reply_text(
        "📁 Send a supported text or code file to analyze it."
    )


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message
    user = update.effective_user

    if not message or not user or not message.text:
        return

    text = message.text.strip()
    user_id = str(user.id)

    if user_id not in USERS:
        USERS[user_id] = {
            "id": user.id,
            "name": user.first_name or "User",
            "username": user.username or ""
        }
        save_data()

    if MODE.get(user_id) == "generate":
        LAST_QUESTION[user_id] = text
        await generate_file(message, user, context)
        MODE[user_id] = "chat"
        return

    await handle_question(message, user, text, context)


async def on_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message

    if not message or not update.effective_user:
        return

    if message.document:
        document = message.document
        filename = document.file_name or "uploaded.txt"

        allowed = (
            ".py", ".txt", ".json", ".html", ".htm",
            ".css", ".js", ".md", ".xml", ".yaml",
            ".yml", ".csv", ".sql", ".sh"
        )

        if not filename.lower().endswith(allowed):
            await message.reply_text(
                "⚠️ Please upload a supported text/code file."
            )
            return

        if document.file_size and document.file_size > MAX_READ_BYTES:
            await message.reply_text(
                "⚠️ Maximum file size is 2 MB."
            )
            return

        await process_uploaded_file(
            update,
            context,
            document.file_id,
            filename
        )
        return

    if message.photo:
        await message.reply_text(
            "📷 Image analysis is not enabled in this version. "
            "Please send a text or code file."
        )


async def callbacks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query

    if not query:
        return

    await query.answer()

    user_id = str(query.from_user.id)
    data = query.data or ""

    if data == "m|home":
        await query.edit_message_text(
            start_text(query.from_user),
            reply_markup=main_menu(),
            parse_mode=ParseMode.HTML
        )

    elif data == "m|help":
        await query.edit_message_text(
            help_text(),
            reply_markup=back_menu(),
            parse_mode=ParseMode.HTML
        )

    elif data == "m|status":
        await query.edit_message_text(
            status_text(),
            reply_markup=back_menu(),
            parse_mode=ParseMode.HTML
        )

    elif data == "m|clear":
        HISTORY[user_id] = []
        save_data()

        await query.edit_message_text(
            "🧹 Chat history cleared.",
            reply_markup=main_menu()
        )

    elif data == "m|chat":
        MODE[user_id] = "chat"

        await query.edit_message_text(
            "💬 <b>AI Chat</b>\n\nSend your question now.",
            reply_markup=back_menu(),
            parse_mode=ParseMode.HTML
        )

    elif data == "m|generate":
        MODE[user_id] = "generate"

        await query.edit_message_text(
            "🛠️ <b>File Generator</b>\n\n"
            "Choose a file type first.",
            reply_markup=gen_type_menu(),
            parse_mode=ParseMode.HTML
        )

    elif data == "m|analyze":
        FILE_ACTION[user_id] = "analyze"

        await query.edit_message_text(
            "📁 Send a supported text or code file for analysis.",
            reply_markup=back_menu()
        )

    elif data == "m|fix":
        FILE_ACTION[user_id] = "fix"

        await query.edit_message_text(
            "🔧 Send a supported text or code file to fix it.",
            reply_markup=back_menu()
        )

    elif data == "m|clear_mode":
        MODE[user_id] = "chat"
        await query.edit_message_text(
            "Chat mode is active.",
            reply_markup=main_menu()
        )

    else:
        await query.edit_message_text(
            "Please choose an option from the menu.",
            reply_markup=main_menu()
        )


async def gen_type_callback(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    query = update.callback_query

    if not query:
        return

    await query.answer()

    user_id = str(query.from_user.id)
    data = query.data or ""

    if not data.startswith("gt|"):
        return

    generation_type = data.split("|", 1)[1].strip().lower()

    allowed_types = {
        "python", "py", "html", "css",
        "javascript", "js", "json", "text", "txt"
    }

    if generation_type not in allowed_types:
        await query.edit_message_text(
            "Unsupported file type.",
            reply_markup=main_menu()
        )
        return

    GEN_TYPE[user_id] = generation_type
    MODE[user_id] = "generate"

    await query.edit_message_text(
        f"🛠️ <b>{html.escape(generation_type.upper())} Generator</b>\n\n"
        "Now send a description of the file you want.",
        parse_mode=ParseMode.HTML,
        reply_markup=back_menu()
    )


async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE
):
    print("TELEGRAM BOT ERROR:", repr(context.error))


async def post_init(application: Application):
    load_data()

    try:
        me = await application.bot.get_me()
        print(f"ZAD AI started: @{me.username}")
    except Exception as exc:
        print("Could not fetch bot details:", repr(exc))


async def post_shutdown(application: Application):
    save_data()
    print("ZAD AI stopped.")


def main():
    if not BOT_TOKEN:
        raise RuntimeError(
            "BOT_TOKEN missing. Add BOT_TOKEN in Render environment variables."
        )

    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .defaults(Defaults(parse_mode=ParseMode.HTML))
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("clear", clear_command))
    application.add_handler(CommandHandler("ping", ping_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("broadcast", broadcast_command))
    application.add_handler(CommandHandler("ask", ask_command))
    application.add_handler(CommandHandler("file", file_command))

    application.add_handler(
        CallbackQueryHandler(
            gen_type_callback,
            pattern=r"^gt\|"
        )
    )

    application.add_handler(
        CallbackQueryHandler(
            callbacks,
            pattern=r"^m\|"
        )
    )

    application.add_handler(
        MessageHandler(
            filters.TEXT & filters.PRIVATE & ~filters.COMMAND,
            on_text
        )
    )

    application.add_handler(
        MessageHandler(
            (filters.Document.ALL | filters.PHOTO) & filters.PRIVATE,
            on_file
        )
    )

    application.add_error_handler(error_handler)

    print("Starting ZAD AI Telegram bot...")
    application.run_polling(
        drop_pending_updates=True,
        allowed_updates=Update.ALL_TYPES
    )


if __name__ == "__main__":
    main()
