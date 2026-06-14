#!/usr/bin/env python3
"""
Gaming Agent - cached ultrafast mode

Goals:
- Background vision cache for speed.
- The agent NEVER speaks just because it analyzed a screen.
- It only answers when the user speaks.
- Live/current-screen questions answer from the latest cache.
- If the cache is empty, it takes one fresh screenshot as fallback.
- General lore/development/world questions skip screen analysis and use web + text model.
- Barge-in: if the user starts speaking while TTS is playing, stop TTS.
- Short noise bursts do not discard pending answers.
"""

import base64
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np
from PIL import Image

import ollama
import sounddevice as sd
from faster_whisper import WhisperModel


# ---------------- ENV HELPERS ----------------

def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)

def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default

def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default

def env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


# ---------------- CONFIG ----------------

VISION_MODEL = env_str("GAG_VISION_MODEL", "qwen2.5vl:3b")
TEXT_MODEL = env_str("GAG_TEXT_MODEL", "qwen3:4b")
GAME_HINT = env_str("GAG_GAME_HINT", "Gothic").strip()
PROMPT_GAME_NAME = env_bool("GAG_PROMPT_GAME_NAME", True)
AUTO_DETECT_STEAM_GAME = env_bool("GAG_AUTO_DETECT_STEAM_GAME", True)
TEXT_TEMPERATURE = env_float("GAG_TEXT_TEMPERATURE", 0.5)
TEXT_TOP_K = env_int("GAG_TEXT_TOP_K", 40)
ONE_PROMPT_MODEL = env_str("GAG_ONE_PROMPT_MODEL", VISION_MODEL)
ONE_PROMPT_NUM_CTX = env_int("GAG_ONE_PROMPT_NUM_CTX", 4096)
ONE_PROMPT_NUM_PREDICT = env_int("GAG_ONE_PROMPT_NUM_PREDICT", 260)
VISION_TEMPERATURE = env_float("GAG_VISION_TEMPERATURE", 0.0)
VISION_TOP_K = env_int("GAG_VISION_TOP_K", 1)

WEB_SEARCH_ENABLED = env_bool("GAG_WEB_SEARCH", True)
WEB_TIMEOUT = env_float("GAG_WEB_TIMEOUT", 3.0)
WEB_MAX_RESULTS = env_int("GAG_WEB_MAX_RESULTS", 2)
WEB_MAX_QUERIES = env_int("GAG_WEB_MAX_QUERIES", 2)
WEB_CACHE_TTL = env_float("GAG_WEB_CACHE_TTL", 300.0)

BACKGROUND_VISION = env_bool("GAG_BACKGROUND_VISION", False)
SCREENSHOT_INTERVAL = env_float("GAG_SCREENSHOT_INTERVAL", 1.75)
MAX_CACHE_AGE = env_float("GAG_MAX_CACHE_AGE", 8.0)
SCREEN_CONTEXT_CACHE_SIZE = env_int("GAG_SCREEN_CONTEXT_CACHE_SIZE", 10)
FRESH_FALLBACK_IF_CACHE_EMPTY = env_bool("GAG_FRESH_FALLBACK_IF_CACHE_EMPTY", False)
SCREEN_ANALYSIS_WAIT_TIMEOUT = env_float("GAG_SCREEN_ANALYSIS_WAIT_TIMEOUT", 30.0)
ANALYSIS_START_AFTER_SECONDS = env_float("GAG_ANALYSIS_START_AFTER_SECONDS", 0.25)

SCREENSHOT_TOOL = env_str("GAG_SCREENSHOT_TOOL", "spectacle")
SAVE_SCREENSHOT_JPG = env_bool("GAG_SAVE_SCREENSHOT_JPG", True)
SCREENSHOT_JPEG_QUALITY = env_int("GAG_SCREENSHOT_JPEG_QUALITY", 80)
CLEAN_TMP_IMAGES_ON_START = env_bool("GAG_CLEAN_TMP_IMAGES_ON_START", True)
IMAGE_MAX_WIDTH = env_int("GAG_IMAGE_MAX_WIDTH", 512)
IMAGE_MAX_HEIGHT = env_int("GAG_IMAGE_MAX_HEIGHT", 288)
IMAGE_JPEG_QUALITY = env_int("GAG_IMAGE_JPEG_QUALITY", 55)

VISION_NUM_CTX = env_int("GAG_VISION_NUM_CTX", 512)
VISION_NUM_PREDICT = env_int("GAG_VISION_NUM_PREDICT", 48)

TEXT_NUM_CTX = env_int("GAG_TEXT_NUM_CTX", 1536)
TEXT_NUM_PREDICT = env_int("GAG_TEXT_NUM_PREDICT", 180)
HISTORY_TURNS = env_int("GAG_HISTORY_TURNS", 6)
HISTORY_MAX_CHARS = env_int("GAG_HISTORY_MAX_CHARS", 1200)
AUTO_ACK_GAME_CORRECTIONS = env_bool("GAG_AUTO_ACK_GAME_CORRECTIONS", True)

WHISPER_SIZE = env_str("GAG_WHISPER_SIZE", "base.en")
WHISPER_DEVICE = env_str("GAG_WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = env_str("GAG_WHISPER_COMPUTE_TYPE", "int8")
WHISPER_BEAM_SIZE = env_int("GAG_WHISPER_BEAM_SIZE", 3)
WHISPER_LANGUAGE = env_str("GAG_WHISPER_LANGUAGE", "en")
WHISPER_VAD_FILTER = env_bool("GAG_WHISPER_VAD_FILTER", True)
WHISPER_NO_SPEECH_THRESHOLD = env_float("GAG_WHISPER_NO_SPEECH_THRESHOLD", 0.75)
WHISPER_MIN_AVG_LOGPROB = env_float("GAG_WHISPER_MIN_AVG_LOGPROB", -1.05)
WHISPER_INITIAL_PROMPT = env_str(
    "GAG_WHISPER_INITIAL_PROMPT",
    "Gaming assistant. Common phrases: what game am I playing, what should I do, what does this say, tell me about the lore, compare this game with another game, what games are similar."
)

VOICE_MODEL = env_str("GAG_VOICE_MODEL", str(Path.home() / "gaming-agent/voices/en_US-amy-medium.onnx"))
TTS_ENABLED = env_bool("GAG_TTS_ENABLED", True)
MAX_VOICE_ADVICE_CHARS = env_int("GAG_MAX_VOICE_ADVICE_CHARS", 650)

SAMPLE_RATE = env_int("GAG_SAMPLE_RATE", 48000)
CHANNELS = env_int("GAG_CHANNELS", 1)
INPUT_DEVICE_RAW = env_str("GAG_INPUT_DEVICE", "default")
INPUT_DEVICE = None if INPUT_DEVICE_RAW.strip().lower() in ("", "default", "none", "-1") else int(INPUT_DEVICE_RAW)

VOICE_THRESHOLD = env_float("GAG_VOICE_THRESHOLD", 0.018)
VOICE_START_THRESHOLD = env_float("GAG_VOICE_START_THRESHOLD", max(VOICE_THRESHOLD, 0.018))
VOICE_CONTINUE_THRESHOLD = env_float("GAG_VOICE_CONTINUE_THRESHOLD", VOICE_THRESHOLD * 0.65)
SPEECH_TRIGGER_SECONDS = env_float("GAG_SPEECH_TRIGGER_SECONDS", 0.18)
PRE_ROLL_SECONDS = env_float("GAG_PRE_ROLL_SECONDS", 0.30)
MIN_SPEECH_SECONDS = env_float("GAG_MIN_SPEECH_SECONDS", 0.55)
SILENCE_END_SECONDS = env_float("GAG_SILENCE_END_SECONDS", 0.75)
MIC_DEBUG = env_bool("GAG_MIC_DEBUG", False)
MIC_LIST_DEVICES = env_bool("GAG_MIC_LIST_DEVICES", False)

WARM_IMAGE_MODEL = env_bool("GAG_WARM_IMAGE_MODEL", True)


# ---------------- STATE ----------------

speech_queue: "queue.Queue[tuple[int, np.ndarray]]" = queue.Queue()

ollama_lock = threading.Lock()
speaking_lock = threading.Lock()
stop_event = threading.Event()
answer_in_progress = threading.Event()

tts_processes_lock = threading.Lock()
tts_processes = []

answer_generation_lock = threading.Lock()
answer_generation = 0

cache_lock = threading.Lock()
latest_screen_cache = ""
latest_screen_cache_time = 0.0
latest_screen_cache_image = ""
screen_context_history = []

history_lock = threading.Lock()
conversation_history = []
runtime_game_context = GAME_HINT.strip() if GAME_HINT else ""

web_cache_lock = threading.Lock()
web_cache = {}


# Speech-triggered screenshot capture.
# When speech starts, we capture one screenshot in parallel while the user talks.
# We only run the slow vision model later if the transcribed question actually needs screen context.
pending_capture_lock = threading.Lock()
pending_capture_by_generation = {}



# ---------------- BARGE-IN HELPERS ----------------

def next_generation() -> int:
    global answer_generation
    with answer_generation_lock:
        answer_generation += 1
        return answer_generation

def current_generation() -> int:
    with answer_generation_lock:
        return answer_generation


# ---------------- UTILS ----------------

def is_bad_vision_output(text: str) -> bool:
    """
    Detect degenerate vision outputs such as:
    !!!!!!!!!!!!!!!!
    ......
    repeated tokens
    prompt echo fragments
    """
    t = clean_for_speech(text)
    if not t:
        return True

    stripped = re.sub(r"\s+", "", t)
    if len(stripped) < 4:
        return True

    # Mostly punctuation / no real words.
    letters = re.findall(r"[A-Za-zΑ-Ωα-ω0-9]", stripped)
    if len(letters) < max(3, len(stripped) * 0.25):
        return True

    # Same character repeated.
    if len(set(stripped)) <= 2 and len(stripped) >= 6:
        return True

    low = t.lower()
    bad_fragments = (
        "game title; scene/menu/dialogue",
        "visible important text/options/objective",
        "danger/status",
    )
    if any(b in low for b in bad_fragments):
        return True

    return False

def is_moondream_model() -> bool:
    return "moondream" in VISION_MODEL.lower()


def cleanup_agent_tmp_images():
    """
    Delete old images created by this agent from /tmp.
    This intentionally does NOT delete every image in /tmp, only gaming_agent_* files.
    """
    if not CLEAN_TMP_IMAGES_ON_START:
        return

    tmp = Path(tempfile.gettempdir())
    patterns = [
        "gaming_agent_screen_*.png",
        "gaming_agent_screen_*.jpg",
        "gaming_agent_screen_*.jpeg",
        "gaming_agent_screen_small.jpg",
        "gaming_agent_warmup.jpg",
    ]

    deleted = 0
    for pat in patterns:
        for p in tmp.glob(pat):
            try:
                p.unlink()
                deleted += 1
            except Exception:
                pass

    print(f"[Tmp] deleted {deleted} old agent image(s) from {tmp}")


def clean_for_speech(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"(?i)^\s*(final answer|answer)\s*[:\-]\s*", "", text).strip()
    text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.replace("`", "")
    text = text.replace("**", "")
    text = text.replace("*", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clamp_answer(text: str) -> str:
    text = clean_for_speech(text)
    if len(text) > MAX_VOICE_ADVICE_CHARS:
        text = text[:MAX_VOICE_ADVICE_CHARS].rsplit(" ", 1)[0].strip() + "..."
    return text

def ollama_chat_safe(model: str, messages: list, options: dict, label: str) -> str:
    start = time.time()
    try:
        with ollama_lock:
            response = ollama.chat(model=model, messages=messages, options=options)
        elapsed = time.time() - start
        text = clean_for_speech(response.get("message", {}).get("content", ""))
        print(f"[{label}] {elapsed:.2f}s")
        return text
    except Exception as e:
        elapsed = time.time() - start
        print(f"[{label} error after {elapsed:.2f}s] {e}")
        return ""


# ---------------- SCREENSHOT / IMAGE ----------------

def take_screenshot() -> str:
    timestamp = int(time.time() * 1000)

    # Capture losslessly first because Spectacle/Wayland screenshot tools are most reliable with PNG.
    raw_png = Path(tempfile.gettempdir()) / f"gaming_agent_screen_{timestamp}.png"

    if SCREENSHOT_TOOL == "grim" and shutil.which("grim"):
        cmd = ["grim", str(raw_png)]
    else:
        spec = shutil.which("spectacle")
        if not spec:
            raise RuntimeError("Spectacle not found. Install with: sudo pacman -S spectacle")
        cmd = [spec, "-b", "-n", "-o", str(raw_png)]

    t0 = time.time()
    p = subprocess.run(
        cmd,
        timeout=4,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    if p.returncode != 0 or not raw_png.exists() or raw_png.stat().st_size < 1024:
        err = (p.stderr or "no stderr").strip()
        raise RuntimeError(f"screenshot failed: {' '.join(cmd)} :: {err}")

    if not SAVE_SCREENSHOT_JPG:
        print(f"[Screenshot] {time.time() - t0:.2f}s png")
        return str(raw_png)

    # Convert full screenshot to JPEG to avoid /tmp growing with large PNGs.
    # This costs a tiny amount of CPU time, but usually saves a lot of disk space.
    jpg_out = Path(tempfile.gettempdir()) / f"gaming_agent_screen_{timestamp}.jpg"
    try:
        img = Image.open(raw_png).convert("RGB")
        img.save(jpg_out, "JPEG", quality=SCREENSHOT_JPEG_QUALITY, optimize=True)
        try:
            raw_png.unlink()
        except Exception:
            pass
        print(f"[Screenshot] {time.time() - t0:.2f}s jpg")
        return str(jpg_out)
    except Exception as e:
        print(f"[Screenshot warning] JPEG conversion failed, using PNG: {e}")
        return str(raw_png)

def prepare_image_for_ollama(path: str) -> str:
    t0 = time.time()
    img = Image.open(path).convert("RGB")
    img.thumbnail((IMAGE_MAX_WIDTH, IMAGE_MAX_HEIGHT))

    out = Path(tempfile.gettempdir()) / f"gaming_agent_screen_small_{int(time.time() * 1000)}_{threading.get_ident()}.jpg"

    img.save(out, "JPEG", quality=IMAGE_JPEG_QUALITY, optimize=True)
    print(f"[ImagePrep] {time.time() - t0:.2f}s")
    return str(out)

def image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")



# ---------------- CONVERSATION HISTORY ----------------

def remember_turn(question: str, answer: str):
    """
    Keep a small in-session memory of useful conversation.
    Explicit user corrections/statements override vision guesses.
    """
    global runtime_game_context

    q = clean_for_speech(question)
    a = clean_for_speech(answer)

    extracted = extract_game_from_question_raw(q) if "extract_game_from_question_raw" in globals() else ""
    if extracted and is_game_context_statement(q):
        runtime_game_context = extracted

    # Do not store greetings/noise/gibberish/punctuation-only model failures.
    if not should_store_in_history(q, a):
        return

    if not runtime_game_context and "you appear to be playing" in a.lower():
        maybe = re.sub(r"(?i).*you appear to be playing\s+", "", a).strip(" .")
        if maybe and len(maybe) <= 60:
            runtime_game_context = maybe

    with history_lock:
        conversation_history.append({"q": q, "a": a, "t": time.time()})
        if len(conversation_history) > HISTORY_TURNS:
            del conversation_history[:-HISTORY_TURNS]

def format_history_for_prompt() -> str:
    with history_lock:
        raw_items = list(conversation_history[-HISTORY_TURNS:])

    items = []
    for item in raw_items:
        q = item.get("q", "")
        a = item.get("a", "")
        if not q:
            continue
        if is_simple_chat_question(q) or looks_like_repeated_gibberish(q) or is_bad_generation_output(a):
            continue
        items.append(item)

    if not items:
        return "None."

    lines = []
    for item in items:
        q = item.get("q", "")
        a = item.get("a", "")
        if q:
            lines.append(f"User: {q}")
        if a:
            lines.append(f"Assistant: {a}")

    text = "\n".join(lines).strip()
    if len(text) > HISTORY_MAX_CHARS:
        text = text[-HISTORY_MAX_CHARS:]
    return text or "None."

def recent_questions_text() -> str:
    with history_lock:
        qs = []
        for item in conversation_history[-HISTORY_TURNS:]:
            q = item.get("q", "")
            a = item.get("a", "")
            if not q:
                continue
            if is_simple_chat_question(q) or looks_like_repeated_gibberish(q) or is_bad_generation_output(a):
                continue
            qs.append(q)
    return " ".join(qs)

def get_runtime_game_context() -> str:
    if GAME_HINT:
        return GAME_HINT
    if runtime_game_context:
        return runtime_game_context
    return ""

def question_looks_like_followup(question: str) -> bool:
    q = clean_query_text(question) if "clean_query_text" in globals() else question.lower()
    words = q.split()

    # Only use history for actual short follow-ups.
    # Do NOT prepend history to normal standalone questions, or search gets polluted.
    explicit = (
        "what about",
        "tell me more",
        "more about",
        "explain more",
        "what else",
        "how so",
    )
    if any(f in q for f in explicit):
        return True

    if len(words) <= 6 and any(w in words for w in ("it", "that", "this", "they", "them", "he", "she")):
        return True

    return False


# ---------------- INTENT ROUTING ----------------

GENERAL_KEYWORDS = (
    "lore", "story", "plot", "world", "setting", "history", "backstory",
    "development", "developer", "studio", "publisher", "release", "making of",
    "behind the scenes", "inspired", "inspiration", "based on", "influence",
    "area", "areas", "region", "faction", "character", "npc", "monster",
    "enemy", "quest", "ending", "old camp", "new camp", "swamp camp", "sleeper",
    "xardas", "gomez", "diego", "milten", "gorn", "lester", "fire mage",
    "water mage", "gothic", "puzzle", "walkthrough", "tactic", "tactics",
    "strategy", "boss", "weakness", "build", "weapon", "armor", "where",
    "how do i", "how to", "guide", "best", "join", "find", "solve",
    "similar", "similarities", "resemblance", "resembles", "resemble",
    "compare", "comparison", "versus", "vs", "like", "games like",
    "other games", "recommend", "recommendation", "alternatives",
    "fact", "facts", "trivia", "explain", "tell me about", "what is",
    "who is", "where is", "when is", "why is", "historical", "real life",
    "real-world", "location", "locations", "city", "cities", "country",
    "countries", "place", "places", "landmark", "landmarks", "map",
    "region", "regions", "biome", "biomes", "area", "areas",
)

LIVE_KEYWORDS = (
    "what game", "game am i", "what am i playing", "playing right now",
    "what's on screen", "whats on screen", "what do you see", "what about now",
    "where am i", "what is happening", "what should i do", "what do i do",
    "what now", "next step", "read this", "what does this say", "which option",
    "look carefully", "current screen", "on screen", "this dialogue",
)

def looks_like_live_question(q: str) -> bool:
    ql = q.lower()
    if "right now" in ql and any(w in ql for w in ("play", "playing", "plane", "laying", "game")):
        return True
    return any(k in ql for k in LIVE_KEYWORDS)

def looks_like_game_name_question(q: str) -> bool:
    ql = q.lower()
    # Be strict here. "What's going on while I'm playing right now?"
    # should be a live-screen question, not a game-title question.
    return (
        "what game" in ql
        or "which game" in ql
        or "game am i" in ql
        or "game are we" in ql
        or "what am i playing" in ql
        or "what are we playing" in ql
    )

def looks_like_general_question(q: str) -> bool:
    ql = q.lower()

    # Knowledge/advice questions should use web/history first, not become screenshot-only.
    general_intent_phrases = (
        "similar", "similarities", "resemblance", "resembles", "resemble",
        "compare", "comparison", "versus", " vs ", "like assassin", "like ",
        "other games", "games like", "recommend", "recommendation", "alternatives",
        "lore", "story", "plot", "world", "setting", "history", "backstory",
        "developer", "studio", "publisher", "inspired", "inspiration",
        "made by", "who made", "who developed",
        "fact", "facts", "trivia", "explain", "tell me about",
        "historical", "real life", "real-world", "where is", "what is",
        "who is", "when is", "why is",
        "location", "locations", "city", "cities", "country", "countries",
        "place", "places", "landmark", "landmarks", "map", "region", "regions",
        "area", "areas", "biome", "biomes",
    )
    if any(p in ql for p in general_intent_phrases):
        return True

    if looks_like_live_question(q):
        return False

    return any(k in ql for k in GENERAL_KEYWORDS)



def should_use_screen_for_question(question: str) -> bool:
    """
    Attach screenshot only when the answer may depend on the player's current
    in-game state: location, objective, visible text, menu, enemies, inventory,
    available actions, or current progression.

    This is intentionally generic. It is NOT tied to one example like quests.
    """
    q = clean_query_text(question)

    screen_phrases = (
        "what should i do", "what do i do", "what now", "next step",
        "where am i", "where should i go", "what is happening",
        "what's on screen", "whats on screen", "what do you see",
        "read this", "what does this say", "which option", "which button",
        "this dialogue", "current screen", "on screen", "look at the screen",
        "look carefully", "what game am i playing", "what game are we playing",
        "what am i playing", "what are we playing",
    )
    if any(p in q for p in screen_phrases):
        return True

    # General context-dependent advice/recommendation questions.
    # Examples:
    # "Any other quest I should look into?"
    # "Which weapon should I use?"
    # "What area should I visit next?"
    # "What build should I try from here?"
    # "Is there anything worth doing around here?"
    advice_words = (
        "should", "look into", "do next", "next", "other", "available",
        "recommend", "worth", "missable", "around here", "from here",
        "at this point", "right now", "nearby", "current"
    )
    context_objects = (
        "quest", "mission", "job", "gig", "contract", "area", "place",
        "location", "weapon", "armor", "item", "build", "skill", "perk",
        "choice", "option", "path", "route", "objective", "enemy", "boss",
        "npc", "vendor", "loot", "side activity", "activity"
    )
    if any(w in q for w in advice_words) and any(o in q for o in context_objects):
        return True

    # Pure knowledge questions should not attach the image because the VLM can over-focus
    # on the screenshot and ignore the web/research question.
    knowledge_phrases = (
        "in general", "lore", "story", "plot", "world", "setting",
        "studio", "developer", "publisher", "made it", "made by",
        "similar", "compare", "comparison", "recommend games", "other games",
        "facts", "trivia", "history", "real life", "historical",
        "who made", "what studio",
    )
    if any(p in q for p in knowledge_phrases):
        return False

    return False


# ---------------- VISION CACHE ----------------

def cache_vision_prompt() -> str:
    game = current_or_detected_game() or GAME_HINT or runtime_game_context or "unknown"
    return f"""/no_think
You are extracting current game state for a gaming assistant.

Autodetected/typed current game:
{game}

Analyze this screenshot and return concise useful context:
- visible game identity if obvious
- current objective / quest / mission text
- location / area / map / journal clues
- door/gate/puzzle/menu/dialogue text if visible
- relevant NPCs, enemies, items, markers, weapons, choices, or available actions
- what the player appears to be trying to do

Do not answer the player. Do not use web knowledge. Just describe useful screen context.
If unclear, say "unclear" plus anything visible.
""".strip()



def analyze_image_for_cache(image_path: str) -> str:
    """
    Compatibility helper for optional/background cache.
    The normal answer path does NOT use this prompt anymore.
    """
    small = prepare_image_for_ollama(image_path)
    img64 = image_to_base64(small)

    answer = ollama_chat_safe(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": cache_vision_prompt(), "images": [img64]}],
        options={
            "num_ctx": VISION_NUM_CTX,
            "num_predict": VISION_NUM_PREDICT,
            "temperature": VISION_TEMPERATURE,
            "top_k": VISION_TOP_K,
        },
        label="VisionCache",
    )

    if not is_bad_vision_output(answer):
        return answer

    print(f"[VisionCache] bad output: {answer!r}")
    return ""

def update_cache_once() -> bool:
    """
    Background cache update:
    screenshot -> vision analysis -> rolling 10-item screen context cache.
    It never speaks and never answers by itself.
    """
    shot = take_screenshot()
    text = analyze_image_for_cache(shot)
    if not text:
        print("[VisionCache] empty; keeping previous cache")
        return False

    remember_screen_context(text, shot)
    print(f"[ScreenCache] {text}")
    return True



def capture_for_generation(gen: int):
    """
    Take exactly one screenshot for this spoken question.
    This starts when speech begins, so capture happens while the user is still talking.
    The slow vision model is NOT run here.
    """
    try:
        print(f"[SpeechCapture] generation {gen}: taking screenshot...")
        shot = take_screenshot()

        with pending_capture_lock:
            pending_capture_by_generation[gen] = {
                "image": shot,
                "time": time.time(),
            }

    except Exception as e:
        print(f"[SpeechCapture error] generation {gen}: {e}")
        with pending_capture_lock:
            pending_capture_by_generation[gen] = {
                "image": "",
                "time": time.time(),
                "error": str(e),
            }


def start_capture_for_generation(gen: int):
    thread = threading.Thread(target=capture_for_generation, args=(gen,), daemon=True)
    thread.start()


def get_capture_for_generation(gen: int, wait_timeout: float = 4.0) -> tuple[str, float]:
    """
    Wait briefly for the screenshot captured when the user started talking.
    Returns (image_path, age_seconds).
    """
    deadline = time.time() + wait_timeout

    while time.time() < deadline:
        with pending_capture_lock:
            item = pending_capture_by_generation.get(gen)

        if item is not None:
            image = (item.get("image") or "").strip()
            age = time.time() - item.get("time", time.time())
            return image, age

        time.sleep(0.03)

    return "", 9999.0


def analyze_captured_screen_for_generation(gen: int) -> str:
    """
    Run the slow vision model only when the question needs the screen.
    """
    image_path, age = get_capture_for_generation(gen, wait_timeout=4.0)
    if not image_path:
        return ""

    text = analyze_image_for_cache(image_path)
    if text:
        global latest_screen_cache, latest_screen_cache_time, latest_screen_cache_image
        with cache_lock:
            latest_screen_cache = text
            latest_screen_cache_time = time.time()
            latest_screen_cache_image = image_path
        print(f"[SpeechScreen] {text}")

    return text or ""


def vision_cache_loop():
    """
    Continuous background screen analyzer.

    Takes and analyzes one screenshot every GAG_SCREENSHOT_INTERVAL seconds,
    keeping the last GAG_SCREEN_CONTEXT_CACHE_SIZE analyses.
    It pauses while the agent is answering so Ollama is not overloaded.
    """
    while not stop_event.is_set():
        if answer_in_progress.is_set():
            time.sleep(0.2)
            continue

        try:
            update_cache_once()
        except Exception as e:
            print(f"[VisionCache error] {e}")

        time.sleep(SCREENSHOT_INTERVAL)


def get_screen_cache() -> tuple[str, float]:
    with cache_lock:
        text = latest_screen_cache.strip()
        age = time.time() - latest_screen_cache_time if latest_screen_cache_time else 9999.0
    return text, age


# ---------------- ANSWERS FROM CACHE ----------------

def infer_game_from_cache(cache: str) -> str:
    if GAME_HINT:
        return GAME_HINT
    if not cache:
        return ""
    first = cache.split(";")[0].strip()
    if first.lower() in ("unknown", "not specified", "game/app", "gaming agent"):
        return ""
    return first

def answer_live_from_speech_screen(question: str, gen: int) -> str:
    """
    Answer current-screen questions from the screenshot captured when speech began.
    Slow vision analysis happens here only for live screen questions.
    """
    if looks_like_game_name_question(question) and GAME_HINT:
        return f"You appear to be playing {GAME_HINT}."

    cache = analyze_captured_screen_for_generation(gen)

    if not cache or is_bad_vision_output(cache):
        if GAME_HINT and looks_like_game_name_question(question):
            return f"You appear to be playing {GAME_HINT}."
        return "I captured the screen, but the vision model gave an unusable result."

    if looks_like_game_name_question(question):
        game = infer_game_from_cache(cache)
        if game:
            return f"You appear to be playing {game}."
        return f"I can see this: {cache}"

    ql = question.lower()

    if any(k in ql for k in ("what should i do", "what do i do", "what now", "next step")):
        parts = [p.strip() for p in cache.split(";") if p.strip()]
        if len(parts) >= 5:
            return f"{parts[0]}: {parts[-1]}"
        return cache

    return cache


# ---------------- GENERAL GAME KNOWLEDGE / WEB ----------------

def clean_query_text(text: str) -> str:
    text = (text or "").lower()

    # Common Whisper/search artifacts that hurt search quality.
    # Keep this generic: no game-specific title corrections.
    replacements = {
        "what's": "what is",
        "whats": "what is",
        "wwhat": "what",
        "hat other": "what other",
        "quits": "quests",
        "quit": "quest",
        "lower": "lore",
        "loar": "lore",
        "plane": "playing",
        "laying": "playing",
        "pplaying": "playing",
        "gimmer": "game",
    }
    for a, b in replacements.items():
        text = text.replace(a, b)

    text = re.sub(r"[^a-z0-9\s:.'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_game_title_guess(title: str) -> str:
    """
    Clean extracted game/topic candidates.

    Generic rule:
    - The autodetected/typed game name is the default current game.
    - Vague references like "the game", "the game I'm playing", "in general",
      or "the lore of the game" resolve to that current game.
    - One-letter junk and polite/search filler are rejected.
    - No game-specific corrections are hardcoded here.
    """
    t = clean_for_speech(title)
    t = re.sub(r"\s+", " ", t).strip(" .,:;!?-'\"")
    if not t:
        return ""

    low = t.lower()
    low = low.replace("pplaying", "playing")

    current = (GAME_HINT or runtime_game_context or "").strip()

    if len(low) <= 2:
        return ""

    current_refs = {
        "the game", "game",
        "the game im playing", "the game i'm playing",
        "game im playing", "game i'm playing",
        "the game i am playing", "game i am playing",
        "the game we are playing", "game we are playing",
        "the game we're playing", "game we're playing",
        "current game", "the current game",
        "in general", "the game in general", "game in general",
    }
    if low in current_refs:
        return current

    junk_contains = (
        "lore of the game",
        "lore of game",
        "about the lore",
        "tell me about",
        "what can you tell",
        "what did you tell",
        "thank you",
        "thanks",
        "in general",
        "the studio that made it",
        "studio that made it",
        "the game i'm playing",
        "the game im playing",
        "game i'm playing",
        "game im playing",
        "what other quests",
        "other quests",
    )
    if any(j in low for j in junk_contains):
        return ""

    # Strip trailing generic modifiers from otherwise valid titles.
    t = re.sub(r"(?i)\s+\bin general\b.*$", "", t).strip(" .,:;!?-'\"")
    t = re.sub(r"(?i)\s+\b(the lore|lore|story|plot|facts|studio|developer|quests?|missions?)\b.*$", "", t).strip(" .,:;!?-'\"")

    if len(t.strip()) <= 2:
        return ""

    final_low = t.lower()
    if final_low in {"general", "in general", "the lore", "lore", "the studio", "studio", "developer", "quest", "quests"}:
        return ""

    return t

def resolve_current_game_reference(question: str) -> str:
    """
    Resolve vague references like 'the game', 'the lore', or 'game I'm playing'
    to the autodetected/typed game.
    """
    q = clean_query_text(question)
    current = (GAME_HINT or runtime_game_context or "").strip()

    if not current:
        return ""

    refs = (
        "the game",
        "current game",
        "game i'm playing",
        "game im playing",
        "game i am playing",
        "game we're playing",
        "game we are playing",
        "the game we're playing",
        "the game we are playing",
        "the current game",
    )
    if any(r in q for r in refs):
        return current

    # If the user asks generic lore/facts/studio/etc. without naming a different game,
    # use the autodetected/typed current game.
    if any(w in q for w in ("lore", "story", "plot", "world", "setting", "studio", "developer", "publisher", "in general")):
        raw = extract_game_from_question_raw(question) if "extract_game_from_question_raw" in globals() else ""
        if not raw:
            return current

    return ""


def is_simple_chat_question(question: str) -> bool:
    """
    Local assistant-control phrases. These should not trigger web, screenshot analysis,
    or long model calls.
    """
    q = clean_query_text(question)
    q = q.strip(" .!?")

    exact = {
        "hello",
        "hi",
        "hey",
        "can you hear me",
        "hello can you hear me",
        "are you there",
        "testing",
        "test",
        "mic test",
        "thank you",
        "thanks",
        "ok thanks",
        "okay thanks",
    }
    if q in exact:
        return True

    patterns = (
        r"^(hello|hi|hey)\b.*\b(can you hear me|are you there)\b",
        r"^(can you hear me|do you hear me|are you listening)\b",
        r"^(this is a test|testing one two|mic test)\b",
    )
    return any(re.search(p, q) for p in patterns)

def answer_simple_chat(question: str) -> str:
    q = clean_query_text(question)
    if "hear" in q or "listening" in q or "test" in q:
        return "Yes, I can hear you."
    if "thank" in q:
        return "You’re welcome."
    return "Hey, I’m listening."

def looks_like_repeated_gibberish(text: str) -> bool:
    """
    Reject repeated filler/hallucinated transcripts such as:
    'Dooboo dooboo dooboo dooboo.'
    """
    q = clean_query_text(text)
    words = re.findall(r"[a-zA-Z']+", q)
    if len(words) < 3:
        return False

    # Same one or two words repeated over and over.
    unique = set(words)
    if len(words) >= 4 and len(unique) <= 2:
        return True

    # One token appears in most of the sentence and the sentence has no question/action words.
    counts = {w: words.count(w) for w in unique}
    most = max(counts.values()) if counts else 0
    intent_words = {
        "what", "where", "when", "why", "how", "who", "which",
        "tell", "explain", "compare", "similar", "lore", "quest",
        "help", "do", "next", "read", "look", "game", "playing",
        "location", "locations", "facts", "advice"
    }
    if most >= 3 and not any(w in intent_words for w in words):
        return True

    return False

def should_store_in_history(question: str, answer: str) -> bool:
    """
    Keep history useful. Do not let mic tests, greetings, or gibberish poison searches.
    """
    if is_simple_chat_question(question):
        return False
    if looks_like_repeated_gibberish(question):
        return False
    if is_bad_generation_output(answer):
        return False
    return True

def is_bad_generation_output(text: str) -> bool:
    """
    Detect punctuation-only/degenerate answers from local models.
    """
    t = clean_for_speech(text)
    if not t:
        return True

    stripped = re.sub(r"\s+", "", t)
    if len(stripped) < 2:
        return True

    alnum = re.findall(r"[A-Za-zΑ-Ωα-ω0-9]", stripped)
    if len(alnum) < max(2, len(stripped) * 0.25):
        return True

    if len(set(stripped)) <= 2 and len(stripped) >= 6:
        return True

    return False



def clean_extracted_game_title(title: str) -> str:
    title = (title or "").strip()
    title = re.sub(r"(?i)\b(please|right now|currently)\b", " ", title)
    title = re.sub(r"(?i)\b(what can you tell.*|tell me.*|what do you know.*|can you tell.*)$", " ", title)
    title = re.sub(r"[,.;?!]+$", "", title).strip(" -:;'\"")
    title = re.sub(r"\s+", " ", title).strip()
    return normalize_game_title_guess(title)


def extract_game_from_question_raw(question: str) -> str:
    """
    Extract only an explicitly named game/topic from the current question.
    Does not fallback to the startup/autodetected game.
    """
    original = (question or "").strip()
    if not original:
        return ""

    cleaned = re.sub(r"[?!]+$", "", original).strip()

    explicit_patterns = [
        r"(?i)\b(?:i'?m|i am|we'?re|we are)\s+playing\s+([^,.?;]+)",
        r"(?i)\b(?:the\s+game\s+is|game\s+is|it'?s|it is|this\s+is)\s+([^,.?;]+)",
        r"(?i)\b(?:no|wrong|incorrect|actually)\s*,?\s*(?:it'?s|it is|the game is|i'?m playing|i am playing)?\s*([^,.?;]+)",
    ]

    for pat in explicit_patterns:
        m = re.search(pat, cleaned)
        if m:
            title = clean_extracted_game_title(m.group(1))
            bad = ("wrong", "again", "with game", "playing", "reply", "general", "thank", "quest")
            if title and len(title) <= 80 and not any(b in title.lower() for b in bad):
                return title

    question_patterns = [
        r"(?i)(?:lore|story|plot|world|setting|history|backstory|facts?|trivia|locations?|places?|map|regions?|cities|landmarks?)\s+(?:of|in|for|about)\s+([^,.?;]+)",
        r"(?i)(?:who made|who developed|what studio made|what studio developed|developer of|studio behind|publisher of)\s+([^,.?;]+)",
        r"(?i)(?:similar to|like|compared to|versus|vs\.?)\s+([^,.?;]+)",
    ]

    for pat in question_patterns:
        m = re.search(pat, cleaned)
        if m:
            title = clean_extracted_game_title(m.group(1))
            if title and 2 <= len(title) <= 80:
                return title

    # Generic "about/in/for X" only if X looks like a real title, not a generic phrase.
    m = re.search(r"(?i)(?:about|for|in)\s+([A-Z][A-Za-z0-9:'& -]{2,80})", cleaned)
    if m:
        title = clean_extracted_game_title(m.group(1))
        if title and 2 <= len(title) <= 80:
            return title

    return ""

def extract_game_from_question(question: str) -> str:
    raw = extract_game_from_question_raw(question)
    if raw and not looks_like_generic_game_reference(raw):
        return raw
    return resolve_current_game_reference(question)


def question_mentions_game(question: str) -> bool:
    raw = extract_game_from_question_raw(question)
    return bool(raw and not looks_like_generic_game_reference(raw))


def is_game_context_statement(question: str) -> bool:
    """
    True when the user is just correcting/setting the game, not asking a real question.
    """
    q = clean_query_text(question) if "clean_query_text" in globals() else question.lower()
    if "?" in question:
        return False
    if re.search(r"\b(i am|i'm|we are|we're)\s+playing\b", q):
        return True
    if re.search(r"\b(the game is|game is|it's|it is)\b", q) and extract_game_from_question(question):
        return True
    if re.search(r"\b(no|wrong|incorrect|actually)\b", q) and extract_game_from_question(question):
        return True
    return False

def looks_like_generic_game_reference(title: str) -> bool:
    """
    Reject extracted 'titles' that are actually generic phrases from the question.
    Examples: 'the game', 'the game and the world', 'the story of the game'.
    """
    t = clean_query_text(title)
    if not t or len(t) <= 2:
        return True

    generic_exact = {
        "game", "the game", "current game", "the current game",
        "this game", "that game", "my game", "our game",
        "the game and the world", "game and the world",
        "the story", "the lore", "the world", "the setting",
        "in general", "general",
    }
    if t in generic_exact:
        return True

    generic_bits = (
        "the game",
        "game i'm playing",
        "game im playing",
        "game i am playing",
        "game we're playing",
        "game we are playing",
        "the story of",
        "the lore of",
        "the world of",
        "story of the game",
        "lore of the game",
        "world of the game",
        "game and the world",
        "story and world",
        "lore and world",
        "the studio that made it",
    )
    return any(bit in t for bit in generic_bits)

def current_or_detected_game() -> str:
    """
    The default target game. Manual/Steam-autodetected game wins over vague extraction.
    """
    return (GAME_HINT or runtime_game_context or "").strip()

def explicit_other_game_from_question(question: str) -> str:
    """
    Return a game/topic only when the question clearly names a non-generic target.
    Otherwise return empty so the current/autodetected game is used.
    """
    raw = extract_game_from_question_raw(question)
    if not raw:
        return ""
    if looks_like_generic_game_reference(raw):
        return ""

    current = current_or_detected_game()
    if current and clean_query_text(raw) == clean_query_text(current):
        return current

    return raw

def get_effective_game_for_question(question: str) -> str:
    """
    Target selection:
    - Use the autodetected/typed current game by default.
    - Only switch targets if the user clearly names another non-generic game/topic.
    - Never let phrases like 'the game and the world' become the game target.
    """
    other = explicit_other_game_from_question(question)
    current = current_or_detected_game()

    if other and (not current or clean_query_text(other) != clean_query_text(current)):
        return other

    if current:
        return current

    return (
        explicit_other_game_from_question(recent_questions_text())
        or infer_game_from_cache(get_screen_cache()[0])
        or "video game"
    )


def strip_polite_filler(text: str) -> str:
    """
    Remove conversational filler from search topics.
    Generic only: no game-specific or prompt-specific hacks.
    """
    q = clean_query_text(text)
    fillers = [
        r"\bhello\b",
        r"\bhi\b",
        r"\bhey\b",
        r"\bthank you so much\b",
        r"\bthank you\b",
        r"\bthanks\b",
        r"\bperfect\b",
        r"\bokay\b",
        r"\bok\b",
        r"\bplease\b",
        r"\bcan you\b",
        r"\bcould you\b",
        r"\btell me\b",
        r"\bi need you to\b",
    ]
    for pat in fillers:
        q = re.sub(pat, " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def looks_like_contextual_recommendation(question: str) -> bool:
    """
    Generic recommendation/advice detector.
    Not tied to a specific game, quest, or example phrase.
    """
    q = clean_query_text(question)

    advice_words = (
        "should", "recommend", "worth", "best", "good", "better",
        "next", "available", "missable", "look into", "do next",
        "from here", "at this point", "right now", "around here",
        "nearby", "which", "what other", "any other"
    )

    target_words = (
        "quest", "mission", "job", "gig", "contract", "area", "location",
        "place", "route", "path", "objective", "choice", "option",
        "weapon", "armor", "item", "build", "skill", "perk", "enemy",
        "boss", "npc", "vendor", "loot", "activity", "side activity",
        "thing", "things"
    )

    return any(w in q for w in advice_words) and any(w in q for w in target_words)

def is_followup_question(question: str) -> bool:
    """
    Generic follow-up detector. Used to resolve pronouns like
    'it', 'that', 'they', etc. from recent conversation.
    """
    q = clean_query_text(question)
    words = q.split()

    followup_phrases = (
        "who was it", "who wrote it", "who made it", "who developed it",
        "who published it", "what about it", "tell me more", "more about",
        "how about that", "what about that", "what was that", "who is he",
        "who is she", "who are they", "where is that", "when was that",
        "what does that mean", "why is that"
    )
    if any(p in q for p in followup_phrases):
        return True

    pronouns = {"it", "that", "this", "they", "them", "he", "she", "there"}
    return len(words) <= 8 and any(w in pronouns for w in words)

def direct_answer_from_history(question: str) -> str:
    """
    Try to answer short follow-up questions from recent assistant answers.
    If not confident, return empty and let the normal vision+web+text flow run.
    """
    if not is_followup_question(question):
        return ""

    q = clean_query_text(question)

    with history_lock:
        items = list(conversation_history[-6:])

    history_text = " ".join(
        clean_for_speech(item.get("a", "")) for item in items
        if item.get("a") and not is_bad_generation_output(item.get("a", ""))
    )
    history_text = re.sub(r"\s+", " ", history_text).strip()
    if not history_text:
        return ""

    patterns = []

    if "written by" in q or "who wrote" in q or "writer" in q:
        patterns.extend([
            r"(?:written by|created by|designed by)\s+([A-Z][A-Za-z .'-]{2,80})",
            r"\b([A-Z][A-Za-z .'-]{2,80})'?s\s+(?:role-playing game|tabletop game|game)",
        ])

    if "developed" in q or "developer" in q or "who made" in q or "made it" in q:
        patterns.extend([
            r"(?:developed by|made by)\s+([A-Z][A-Za-z0-9 .&'-]{2,80})",
        ])

    if "published" in q or "publisher" in q:
        patterns.extend([
            r"(?:published by)\s+([A-Z][A-Za-z0-9 .&'-]{2,80})",
        ])

    if "based on" in q:
        patterns.extend([
            r"(?:based on)\s+([^.;]{3,120})",
        ])

    for pat in patterns:
        m = re.search(pat, history_text)
        if m:
            ans = m.group(1).strip(" .,:;")
            if ans:
                if "written by" in q or "who wrote" in q:
                    return f"It was written/created by {ans}."
                if "developed" in q or "developer" in q or "who made" in q or "made it" in q:
                    return f"It was developed by {ans}."
                if "published" in q or "publisher" in q:
                    return f"It was published by {ans}."
                if "based on" in q:
                    return f"It was based on {ans}."
                return ans

    return ""

def search_context_for_followup(question: str) -> str:
    """
    Tiny generic search hint for follow-up questions.
    Never dump full conversation history into web queries.
    """
    q = clean_query_text(question)

    if "written by" in q or "who wrote" in q or "writer" in q:
        return "writer creator author"
    if "developed" in q or "developer" in q or "who made" in q or "made it" in q:
        return "developer studio"
    if "published" in q or "publisher" in q:
        return "publisher"
    if "based on" in q:
        return "based on source material"
    if "who" in q:
        return "who"
    if "where" in q:
        return "location"
    if "when" in q:
        return "release date year"
    return ""

def remember_screen_context(text: str, image_path: str = ""):
    """
    Store recent vision/context results so follow-up questions can use previous images.
    """
    global latest_screen_cache, latest_screen_cache_time, latest_screen_cache_image

    text = clean_for_speech(text or "")
    if not text or is_bad_generation_output(text):
        return

    with cache_lock:
        latest_screen_cache = text
        latest_screen_cache_time = time.time()
        latest_screen_cache_image = image_path or latest_screen_cache_image
        screen_context_history.append({
            "t": time.time(),
            "text": text,
            "image": image_path or "",
        })
        # Keep the last N analyses for follow-ups without unlimited prompt bloat.
        if len(screen_context_history) > SCREEN_CONTEXT_CACHE_SIZE:
            del screen_context_history[:-SCREEN_CONTEXT_CACHE_SIZE]

def format_screen_context_history(max_chars: int = 1400) -> str:
    """
    Return recent screen analyses for prompt context.
    The newest/current analysis is also passed separately; this is for prior images.
    """
    with cache_lock:
        items = list(screen_context_history[-SCREEN_CONTEXT_CACHE_SIZE:])

    if not items:
        return "None."

    lines = []
    now = time.time()
    for item in items:
        age = int(now - item.get("t", now))
        txt = clean_for_speech(item.get("text", ""))
        if txt:
            lines.append(f"{age}s ago: {txt}")

    out = "\n".join(lines).strip()
    if len(out) > max_chars:
        out = out[-max_chars:]
    return out or "None."

def extract_screen_search_terms(screen_context: str, max_terms: int = 4) -> list[str]:
    """
    Pull useful search terms from vision results:
    objectives, quoted text, locations, and short capitalized chunks.
    Generic; not tied to one game.
    """
    if not screen_context:
        return []

    candidates = []

    # Quoted objective/menu/location text is usually gold.
    for m in re.finditer(r'["“”]([^"“”]{3,70})["“”]', screen_context):
        candidates.append(m.group(1).strip())

    # Common objective wording: current objective reads: Ascend Mar Guran.
    objective_patterns = [
        r"objective(?: or quest text)?(?: visible)?(?: reads| indicates| is)?[:\s]+([A-Z][A-Za-z0-9' -]{3,80})",
        r"quest text(?: visible)?(?: reads| indicates| is)?[:\s]+([A-Z][A-Za-z0-9' -]{3,80})",
        r"location(?: clues)?(?: include| is|:)?\s+([A-Z][A-Za-z0-9' -]{3,80})",
    ]
    for pat in objective_patterns:
        for m in re.finditer(pat, screen_context):
            val = m.group(1).strip(" .,:;")
            # Stop at sentence-ish boundary words if regex over-captures.
            val = re.split(r"\b(?:There|The player|In terms|No visible|suggesting|with)\b", val)[0].strip(" .,:;")
            if val:
                candidates.append(val)

    # Capitalized chunks like Mar Guran, Ancient Hallways, Ruined Tower.
    for m in re.finditer(r"\b([A-Z][A-Za-z0-9']+(?:\s+[A-Z][A-Za-z0-9']+){1,4})\b", screen_context):
        chunk = m.group(1).strip()
        if 3 <= len(chunk) <= 60:
            candidates.append(chunk)

    bad = {
        "The", "There", "This", "Current", "Location", "Vision", "Fatekeeper",
        "The game", "In terms", "No visible", "The player"
    }

    clean = []
    seen = set()
    for c in candidates:
        c = re.sub(r"\s+", " ", c).strip(" .,:;!?-'\"")
        if not c or c in bad:
            continue
        low = clean_query_text(c)
        if low in seen:
            continue
        # Avoid generic descriptions.
        if any(x in low for x in ("fantasy role playing", "health bar", "bottom right", "left side")):
            continue
        seen.add(low)
        clean.append(c)
        if len(clean) >= max_terms:
            break

    return clean

def cached_screen_search_terms(max_terms: int = 4) -> list[str]:
    with cache_lock:
        items = list(screen_context_history[-SCREEN_CONTEXT_CACHE_SIZE:])

    merged = []
    seen = set()
    for item in reversed(items):
        for term in extract_screen_search_terms(item.get("text", ""), max_terms=max_terms):
            low = clean_query_text(term)
            if low and low not in seen:
                seen.add(low)
                merged.append(term)
                if len(merged) >= max_terms:
                    return merged
    return merged

def build_search_queries(question: str, screen_context: str = "") -> list[str]:
    """
    Build focused searches from the current/autodetected game + actual topic + vision context.

    Core rule:
    Search query must start from the autodetected/typed game name unless the user
    clearly names another game. Vague phrases like "the game", "the story", or
    "the game and the world" are not allowed to become the search target.

    Screen context is used for location/objective terms such as "Mar Guran".
    """
    q = clean_query_text(question)

    current = current_or_detected_game()
    other = explicit_other_game_from_question(question)
    if other and (not current or clean_query_text(other) != clean_query_text(current)):
        base = clean_query_text(other)
    else:
        base = clean_query_text(current or "video game")

    if not base or looks_like_generic_game_reference(base):
        base = clean_query_text(current or GAME_HINT or runtime_game_context or "video game")

    # Pull high-value terms from current screenshot; if not enough, use prior screen cache.
    screen_terms = extract_screen_search_terms(screen_context)
    if not screen_terms:
        screen_terms = cached_screen_search_terms()
    screen_topic = " ".join(screen_terms[:3]).strip()

    topic = strip_polite_filler(question)

    filler_patterns = [
        r"\bwhat can you tell me about\b",
        r"\bwhat can you tell\b",
        r"\bwhat did you tell about\b",
        r"\btell me about\b",
        r"\bcan you tell me about\b",
        r"\bi know that\b",
        r"\bperfect\b",
        r"\bcontinue\b",
        r"\bthe game i'?m playing\b",
        r"\bthe game im playing\b",
        r"\bthe current game\b",
        r"\bcurrent game\b",
        r"\bthe game\b",
        r"\bin general\b",
        r"\bright now\b",
        r"\byeah\b",
        r"\bbut\b",
    ]
    for pat in filler_patterns:
        topic = re.sub(pat, " ", topic, flags=re.IGNORECASE)

    # Remove target game words from topic so we don't duplicate them.
    if base and base != "video game":
        for part in base.split():
            if len(part) >= 3:
                topic = re.sub(rf"\b{re.escape(part)}\b", " ", topic)

    topic = re.sub(r"\s+", " ", topic).strip(" .,:;!?-'\"")

    practical = any(w in q for w in ("stuck", "open", "unlock", "door", "gate", "puzzle", "lever", "switch", "key", "access", "cannot", "can't"))
    screen_prefix = f"{screen_topic} " if screen_topic else ""

    if practical:
        topic = f"{screen_prefix}door open walkthrough".strip()
    elif any(p in q for p in ("what game am i playing", "where should i go", "where do i go", "go next", "what should i do", "what do i do", "next step")):
        topic = f"{screen_prefix}current objective walkthrough".strip()
    elif any(w in q for w in ("lore", "story", "plot", "world", "setting", "history", "backstory")):
        topic = "lore story world setting"
    elif any(w in q for w in ("studio", "developer", "developed", "made", "created", "who made", "publisher", "published")):
        topic = "developer studio publisher"
    elif any(w in q for w in ("written by", "who wrote", "writer", "author", "creator", "created by")):
        topic = "writer creator author"
    elif any(w in q for w in ("location", "locations", "area", "areas", "region", "map", "where", "place", "places")):
        topic = f"{screen_prefix}locations places map regions".strip()
    elif looks_like_contextual_recommendation(question):
        topic = f"{screen_prefix}recommended guide best options available".strip()
    elif not topic:
        topic = f"{screen_prefix}overview lore gameplay developer".strip()

    queries = []

    if practical:
        # Use screen terms first; this is how "door" becomes "Mar Guran door".
        queries.extend([
            f"{base} {topic}",
            f"{base} {screen_prefix}how to open door",
            f"{base} {screen_prefix}door guide",
            f"{base} {screen_prefix}walkthrough",
        ])
    elif any(p in q for p in ("what game am i playing", "where should i go", "where do i go", "go next", "what should i do", "what do i do", "next step")):
        queries.extend([
            f"{base} {topic}",
            f"{base} {screen_prefix}locations map walkthrough",
            f"{base} {screen_prefix}quest guide",
            f"{base} beginner guide objective marker",
        ])
    elif any(w in q for w in ("lore", "story", "plot", "world", "setting", "history", "backstory")):
        queries.extend([
            f"{base} lore",
            f"{base} story world setting",
            f"{base} wiki lore story setting",
            f"{base} plot summary world story",
        ])
    elif any(w in q for w in ("studio", "developer", "developed", "made", "created", "who made", "publisher", "published")):
        queries.extend([
            f"{base} developer studio publisher",
            f"{base} developed by publisher",
            f"{base} wiki developer publisher",
            f"{base} steam developer publisher",
        ])
    elif any(w in q for w in ("written by", "who wrote", "writer", "author", "creator", "created by")):
        queries.extend([
            f"{base} writer creator author",
            f"{base} created by written by",
            f"{base} source material creator",
            f"{base} wiki creator writer",
        ])
    elif looks_like_contextual_recommendation(question):
        queries.extend([
            f"{base} {topic} recommended guide",
            f"{base} {topic} best options available",
            f"{base} {topic} wiki guide",
            f"{base} {topic} tips",
        ])
    elif any(w in q for w in ("similar", "similarities", "resemblance", "resembles", "resemble", "compare", "comparison", "versus", " vs ", "other games", "games like", "recommend", "alternatives")):
        queries.extend([
            f"{base} {topic} comparison similarities",
            f"{base} {topic} games like",
            f"{base} similar games recommendations",
            f"{base} gameplay comparison",
        ])
    elif any(w in q for w in ("area", "areas", "region", "map", "location", "locations", "where", "city", "cities", "country", "countries", "place", "places", "landmark", "landmarks")):
        queries.extend([
            f"{base} {topic} locations places map regions",
            f"{base} {topic} wiki locations",
            f"{base} real world locations setting",
            f"{base} location guide map",
        ])
    elif any(w in q for w in ("fact", "facts", "trivia", "explain", "what is", "who is", "when is", "why is", "historical", "real life", "real-world")):
        queries.extend([
            f"{base} {topic}",
            f"{base} {topic} facts explanation",
            f"{base} {topic} wiki overview",
            f"{base} {topic} guide",
        ])
    else:
        queries.extend([
            f"{base} {topic}",
            f"{base} {topic} guide",
            f"{base} {topic} wiki",
            f"{base} {topic} reddit",
        ])

    seen = set()
    out = []
    for item in queries:
        item = re.sub(r"\s+", " ", item).strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)

    return out[:WEB_MAX_QUERIES]

def web_search_snippets(query: str | list[str]) -> str:
    if not WEB_SEARCH_ENABLED:
        return ""

    try:
        from ddgs import DDGS
    except Exception:
        print("[Web] ddgs not installed. Install with: pip install ddgs")
        return ""

    queries = query if isinstance(query, list) else [query]
    queries = queries[:WEB_MAX_QUERIES]
    cache_key = tuple(queries)

    now = time.time()
    with web_cache_lock:
        cached = web_cache.get(cache_key)
        if cached:
            ts, value = cached
            if now - ts <= WEB_CACHE_TTL:
                print(f"[Web] cache hit, {len(value.splitlines())} results")
                return value

    t0 = time.time()
    results = []
    seen_urls = set()

    try:
        with DDGS(timeout=WEB_TIMEOUT) as ddgs:
            for q in queries:
                print(f"[Web] query: {q}")
                try:
                    for r in ddgs.text(q, max_results=WEB_MAX_RESULTS):
                        title = (r.get("title") or "").strip()
                        body = (r.get("body") or "").strip()
                        href = (r.get("href") or "").strip()

                        if href and href in seen_urls:
                            continue
                        if href:
                            seen_urls.add(href)

                        if title or body:
                            results.append(f"{title}: {body} ({href})")

                        if len(results) >= WEB_MAX_RESULTS * WEB_MAX_QUERIES:
                            break
                except Exception as e:
                    print(f"[Web query error] {q}: {e}")

                if len(results) >= WEB_MAX_RESULTS * WEB_MAX_QUERIES:
                    break

        print(f"[Web] {time.time() - t0:.2f}s, {len(results)} results")
    except Exception as e:
        print(f"[Web error] {e}")

    value = "\n".join(results[:WEB_MAX_RESULTS * WEB_MAX_QUERIES])
    with web_cache_lock:
        web_cache[cache_key] = (time.time(), value)
    return value



def extract_relevant_lines_from_web(web: str, max_lines: int = 5) -> str:
    """
    Lightweight deterministic fallback when the local LLM refuses to answer.
    Filters generic/irrelevant search results when possible.
    """
    if not web:
        return ""

    current = clean_query_text(GAME_HINT or runtime_game_context or "")
    current_tokens = [w for w in current.split() if len(w) >= 4]

    lines = []
    for raw in web.splitlines():
        line = raw.strip()
        if not line:
            continue

        low = clean_query_text(line)

        # Avoid generic "what is lore in games" junk when a current game is known.
        if current_tokens and not any(tok in low for tok in current_tokens):
            continue

        # Strip URL and trim.
        line = re.sub(r"\s*\(https?://[^)]*\)\s*$", "", line).strip()
        line = re.sub(r"\s+", " ", line)

        if line and line not in lines:
            lines.append(line)

        if len(lines) >= max_lines:
            break

    if not lines:
        for raw in web.splitlines():
            line = raw.strip()
            if not line:
                continue
            line = re.sub(r"\s*\(https?://[^)]*\)\s*$", "", line).strip()
            line = re.sub(r"\s+", " ", line)
            if line and line not in lines:
                lines.append(line)
            if len(lines) >= max_lines:
                break

    return " ".join(lines)



def extract_candidate_names_from_web(web: str, max_names: int = 8) -> list[str]:
    """
    Extract likely quest/mission/location/item names from search snippets.
    Generic enough for games: quoted names, title fragments, and capitalized noun chunks.
    """
    if not web:
        return []

    candidates = []

    # Quoted titles are often quest names.
    for m in re.finditer(r'["“”]([^"“”]{3,70})["“”]', web):
        name = m.group(1).strip()
        if name:
            candidates.append(name)

    # Pull from result titles before ":".
    for raw in web.splitlines():
        title = raw.split(":", 1)[0].strip()
        title = re.sub(r"\s*[-|]\s*(Wiki|Guide|IGN|Game8|Game Rant|Reddit|YouTube|Polygon|Eurogamer|Rock Paper Shotgun|GameSpot).*$", "", title, flags=re.IGNORECASE)
        title = re.sub(r"(?i)\b(cyberpunk 2077|walkthrough|guide|wiki|side missions?|side quests?|ending guide|quest order|story-focused|ultimate|best|all)\b", " ", title)
        title = re.sub(r"\s+", " ", title).strip(" -:|")
        if title and 3 <= len(title) <= 70:
            candidates.append(title)

    # Capitalized chunks, useful for names like Panam, Judy, Chippin' In.
    for m in re.finditer(r"\b([A-Z][A-Za-z0-9']+(?:\s+[A-Z][A-Za-z0-9']+){0,4})\b", web):
        chunk = m.group(1).strip()
        if 3 <= len(chunk) <= 60:
            candidates.append(chunk)

    bad_words = {
        "The", "This", "That", "These", "Those", "Cyberpunk", "Cyberpunk 2077",
        "Wikipedia", "Reddit", "Guide", "Walkthrough", "Side", "Quests",
        "Missions", "Ending", "Game", "Games", "Story", "Release", "Review",
        "Gameplay", "More", "How To Get", "Ultimate", "Best", "All"
    }

    clean = []
    seen = set()
    for c in candidates:
        c = re.sub(r"\s+", " ", c).strip(" .,:;!?-'\"")
        if not c or c in bad_words:
            continue
        low = c.lower()
        if low in seen:
            continue
        if any(b.lower() == low for b in bad_words):
            continue
        # Avoid full generic result titles.
        if len(c.split()) > 6:
            continue
        seen.add(low)
        clean.append(c)
        if len(clean) >= max_names:
            break

    return clean

def fallback_for_failed_screen_question(question: str, game: str) -> str:
    """
    Generic fallback when the final model fails after vision has already run.
    Uses the known/autodetected game instead of pretending nothing is known.
    """
    q = clean_query_text(question)
    game = (game or GAME_HINT or runtime_game_context or "the current game").strip()

    asks_game = any(p in q for p in (
        "what game am i playing",
        "what game are we playing",
        "what am i playing",
        "which game am i playing",
    ))
    asks_direction = any(p in q for p in (
        "where should i go",
        "where do i go",
        "where next",
        "go next",
        "what should i do",
        "what do i do",
        "what now",
        "next step",
    ))

    if asks_game and asks_direction:
        return f"You are playing {game}. The screen analysis found some current-state info, but the final text model failed, so I cannot confidently synthesize the next step. Use the visible objective/marker as the safest next direction."
    if asks_game:
        return f"You are playing {game}."
    if asks_direction:
        return f"The current game is {game}. The final text model failed, so I cannot confidently synthesize the route; follow the visible objective/marker or journal entry."
    return ""

def wants_actionable_help(question: str) -> bool:
    """
    Generic detector for practical "how do I do this?" questions.
    """
    q = clean_query_text(question)
    action_words = (
        "how", "what can i do", "what do i do", "how do i",
        "stuck", "open", "unlock", "enter", "get past", "solve",
        "use", "activate", "press", "find", "reach", "access",
        "door", "gate", "puzzle", "lever", "switch", "key"
    )
    return any(w in q for w in action_words)

def parse_web_results(web: str) -> list[dict]:
    """
    Parse lines of 'Title: body (url)' into structured records.
    We use bodies for steps; titles are only hints.
    """
    results = []
    if not web:
        return results

    for raw in web.splitlines():
        line = raw.strip()
        if not line:
            continue

        url = ""
        m = re.search(r"\((https?://[^)]*)\)\s*$", line)
        if m:
            url = m.group(1)
            line = line[:m.start()].strip()

        title = ""
        body = line
        if ": " in line:
            title, body = line.split(": ", 1)
        elif ":" in line:
            title, body = line.split(":", 1)

        title = re.sub(r"\s+", " ", title).strip()
        body = re.sub(r"\s+", " ", body).strip()

        if title or body:
            results.append({"title": title, "body": body, "url": url})

    return results

def split_snippet_sentences(web: str) -> list[str]:
    """
    Convert search result bodies into candidate sentences.
    Important: result titles are NOT treated as steps.
    """
    if not web:
        return []

    results = parse_web_results(web)
    chunks = []

    if results:
        for r in results:
            body = r.get("body", "")
            if body:
                chunks.append(body)
    else:
        cleaned = re.sub(r"\s*\(https?://[^)]*\)\s*", " ", web)
        chunks.append(cleaned)

    out = []
    seen = set()
    for chunk in chunks:
        chunk = re.sub(r"\b[A-Z][a-z]{2} \d{1,2}, \d{4}\s*[·-]\s*", " ", chunk)
        chunk = re.sub(r"\s+", " ", chunk).strip()
        parts = re.split(r"(?<=[.!?])\s+|(?:\s+[·•]\s+)", chunk)

        for p in parts:
            p = p.strip(" .,:;|")
            if not p:
                continue
            low = clean_query_text(p)
            if low in seen:
                continue
            seen.add(low)
            out.append(p)

    return out


def extract_actionable_steps_from_web(question: str, web: str, max_steps: int = 4) -> list[str]:
    """
    Extract practical steps from web result bodies.
    Titles are not accepted as steps, because titles like
    'How To Open The Door...' are not instructions.
    """
    q = clean_query_text(question)
    sentences = split_snippet_sentences(web)

    step_markers = (
        "to open", "to unlock", "to enter", "to access", "to get",
        "you have to", "you need to", "you must", "you should",
        "first", "then", "after that", "next", "press", "hold",
        "switch to", "use", "activate", "interact", "requires",
        "can be opened", "opened from", "from elsewhere", "key",
        "lever", "mechanism", "spell", "ability", "telekinesis",
        "go around", "return later", "alternate route", "other side"
    )

    title_only_bad = (
        "how to open", "guide", "walkthrough", "wiki", "tips", "route"
    )

    q_tokens = [w for w in q.split() if len(w) >= 4]
    scored = []

    for s in sentences:
        raw = re.sub(r"\s+", " ", s).strip(" .,:;")
        low = clean_query_text(raw)
        if len(low) < 18:
            continue

        # Drop pure title-like fragments.
        if any(low.startswith(bad) for bad in title_only_bad) and not any(marker in low for marker in ("you have to", "you need to", "switch to", "press", "use", "can be opened", "opened from")):
            continue

        score = 0
        for marker in step_markers:
            if marker in low:
                score += 4

        for tok in q_tokens:
            if tok in low:
                score += 1

        # Strong boost for concrete instruction-y terms.
        if any(x in low for x in ("telekinesis", "opened from elsewhere", "switch to", "press", "use", "lever", "key", "mechanism")):
            score += 6

        if any(bad in low for bad in ("privacy policy", "cookie", "subscribe", "review release gameplay", "start fatekeeper the right way")):
            score -= 8

        if score > 0:
            scored.append((score, raw))

    scored.sort(key=lambda x: x[0], reverse=True)

    steps = []
    seen = set()
    for _score, s in scored:
        s = re.sub(r"\s+", " ", s).strip(" .,:;")

        # Remove date prefix, if any.
        s = re.sub(r"^[A-Z][a-z]{2} \d{1,2}, \d{4}\s*[·-]\s*", "", s).strip()

        low = clean_query_text(s)
        if low in seen:
            continue
        seen.add(low)

        if len(s) > 240:
            s = s[:240].rsplit(" ", 1)[0] + "..."

        steps.append(s)
        if len(steps) >= max_steps:
            break

    return steps

def synthesize_actionable_fallback(question: str, web: str, game: str) -> str:
    """
    Deterministic last-resort synthesis for practical questions.
    Never dumps raw search results or page titles.
    """
    if not wants_actionable_help(question):
        return ""

    steps = extract_actionable_steps_from_web(question, web, max_steps=4)
    if not steps:
        return ""

    q = clean_query_text(question)
    game = game or current_or_detected_game() or "the game"

    if any(w in q for w in ("door", "gate", "open", "unlock")):
        intro = f"For {game}, the useful web info suggests this door/gate is solved with a specific mechanic or route, not by simply pushing it from the front."
    elif any(w in q for w in ("puzzle", "solve")):
        intro = f"For {game}, the useful web info points to these puzzle steps."
    else:
        intro = f"For {game}, the useful web info points to this."

    # Merge overlapping fragments so TTS does not sound like a result dump.
    cleaned_steps = []
    for s in steps:
        s = s.strip(" .")
        if not s:
            continue
        # Avoid repeating the page title in the answer.
        if clean_query_text(s).startswith("how to open"):
            continue
        cleaned_steps.append(s)

    if not cleaned_steps:
        return ""

    if len(cleaned_steps) == 1:
        return f"{intro} {cleaned_steps[0]}."

    return f"{intro} Try this: " + " ".join(f"{i+1}. {step}." for i, step in enumerate(cleaned_steps[:4]))

def synthesized_fallback_answer(question: str, web: str, game: str, used_screen: bool) -> str:
    """
    Last-resort answer that is still assistant-like.
    It should never paste raw search results. It extracts useful names/options
    when possible, otherwise gives a concise uncertainty message.
    """
    screen_answer = fallback_for_failed_screen_question(question, game) if used_screen else ""
    if screen_answer:
        return screen_answer

    actionable = synthesize_actionable_fallback(question, web, game)
    if actionable:
        return actionable

    names = extract_candidate_names_from_web(web, max_names=6)
    wants_recommendation = looks_like_contextual_recommendation(question)

    if wants_recommendation:
        if names:
            picks = names[:4]
            if len(picks) == 1:
                return f"For {game}, I’d look into {picks[0]}. I could not verify your exact current progression, so check whether it is available in your journal/map."
            return f"For {game}, useful options to check are {', '.join(picks[:-1])}, and {picks[-1]}. I could not verify exact availability, so pick the ones currently visible in your journal/map."
        return f"I could not extract specific names from the web results. For {game}, check your journal/map for currently available objectives, side activities, vendors, items, or locations, then ask me about any name you see."

    relevant = extract_relevant_lines_from_web(web, max_lines=3)
    if relevant:
        return f"I found relevant info for {game}, but the model failed to synthesize it cleanly. The most useful bits are: {relevant}"

    if used_screen:
        return "I looked at the screen, but the model result was unusable and the web results were not specific enough."
    return "I could not answer that reliably from the available web results."


def direct_fact_from_snippets(question: str, web: str) -> str:
    """
    Generic, not game-specific.
    Handles common factual questions by extracting from search snippets.
    """
    q = clean_query_text(question)
    if not web:
        return ""

    # For simple "who made/developed/studio" questions, snippets often contain the answer plainly.
    if any(w in q for w in ("studio", "developer", "developed", "made", "created", "who made")):
        # Patterns like "Developer(s) Piranha Bytes" or "... developed by Piranha Bytes"
        patterns = [
            r"developer(?:\(s\))?\s*[:\-]?\s*([A-Z][A-Za-z0-9 &.,'/-]{2,80})",
            r"developed by\s+([A-Z][A-Za-z0-9 &.,'/-]{2,80})",
            r"made by\s+([A-Z][A-Za-z0-9 &.,'/-]{2,80})",
            r"created by\s+([A-Z][A-Za-z0-9 &.,'/-]{2,80})",
        ]
        for pat in patterns:
            m = re.search(pat, web, flags=re.IGNORECASE)
            if m:
                ans = m.group(1).strip()
                ans = re.split(r"[.;()\[]", ans)[0].strip()
                if ans:
                    return f"It was developed by {ans}."

    return ""


def looks_like_prompt_echo(answer: str) -> bool:
    a = (answer or "").lower()
    bad = (
        "answer the user directly",
        "web/reference snippets",
        "web snippets",
        "do not describe",
        "do not invent",
        "final answer only",
        "if the snippets are weak",
        "for simple factual questions",
        "you are a concise",
        "you are a general gaming assistant",
        "use this order",
        "player-provided/current game",
        "recent conversation",
        "online snippets",
        "screenshot:",
    )
    return any(b in a for b in bad)

def answer_general_game_question(question: str) -> str:
    game = get_effective_game_for_question(question)
    queries = build_search_queries(question, "")

    print(f"[Web] game target: {game}")
    print(f"[Web] searching {len(queries)} focused queries for: {question}")
    web = web_search_snippets(queries)

    direct = direct_fact_from_snippets(question, web)
    if direct:
        print("[WebFacts] answered directly from snippets")
        return direct

    history = format_history_for_prompt()

    prompt = f"""
You are a concise, general-purpose gaming assistant.

Current game/context:
{game}

Recent conversation:
{history}

User question:
{question}

Web snippets:
{web if web else "None."}

Answer the user's question directly.
Use the snippets when they are relevant.
If the user asks for similarities, comparisons, recommendations, lore, development, or gameplay context, answer that topic directly.
Do not describe the current screenshot unless the user explicitly asked about the current screen.
Do not claim certainty when the snippets are weak.
Keep the answer natural and short, around 3 to 6 sentences.
""".strip()

    answer = ollama_chat_safe(
        model=TEXT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={
            "num_ctx": TEXT_NUM_CTX,
            "num_predict": TEXT_NUM_PREDICT,
            "temperature": TEXT_TEMPERATURE,
            "top_k": TEXT_TOP_K,
        },
        label="Text",
    )

    if answer and not looks_like_prompt_echo(answer) and "too little reliable information" not in answer.lower():
        return answer

    snippet_fallback = extract_relevant_lines_from_web(web, max_lines=4)
    if snippet_fallback:
        print("[WebFallback] using snippets directly")
        return snippet_fallback

    return "I could not find enough reliable information online for that."



# ---------------- ONE GENERAL PROMPT ANSWERING ----------------


def summarize_screen_context_for_question(question: str, image_b64: str, game: str) -> str:
    """
    Always use vision to extract current state from the screenshot.
    It should not answer the player; it should return context for the final text model.
    """
    if not image_b64:
        return ""

    prompt = f"""/no_think
You are extracting current game state for a final-answer assistant.

Autodetected/typed current game:
{game}

Player question:
{question}

Analyze the screenshot and return concise current-state context:
- visible game identity if obvious
- current objective / quest / mission text
- location / area / map / journal clues if visible
- menu/dialogue text if visible
- relevant NPCs, enemies, items, markers, weapons, choices, or available actions
- anything that helps answer the player's question

Do not answer the player directly. Do not give final advice. Just provide useful screen context.
If the screenshot is unclear, say "unclear" and mention anything you can still identify.
""".strip()

    msg = {"role": "user", "content": prompt, "images": [image_b64]}

    summary = ollama_chat_safe(
        model=ONE_PROMPT_MODEL,
        messages=[msg],
        options={
            "num_ctx": max(1024, VISION_NUM_CTX),
            "num_predict": 140,
            "temperature": VISION_TEMPERATURE,
            "top_k": VISION_TOP_K,
        },
        label="VisionContext",
    )

    if not summary or is_bad_generation_output(summary) or looks_like_prompt_echo(summary):
        return ""

    return summary

def build_one_general_prompt(question: str, game: str, history: str, web: str, screen_context: str = "", screen_history: str = "") -> str:
    """
    One general final-analysis prompt.

    Pipeline policy:
    - Vision runs first and produces screen/current-state context.
    - Web runs and produces external snippets.
    - This text model produces the final answer using both.
    """
    return f"""/no_think
You are a general gaming assistant.

Autodetected/typed current game:
{GAME_HINT or runtime_game_context or "unknown"}

Target game or topic for this question:
{game}

Recent useful conversation history:
{history}

Vision result / current screen analysis:
{screen_context if screen_context else "None."}

Recent previous screen analyses:
{screen_history if screen_history else "None."}

Online snippets:
{web if web else "None."}

Search note:
Searches were generated using the current/autodetected game as the base target unless the player clearly named a different game.

Player question:
{question}

Rules:
1. Use the autodetected/typed current game as the default game. Treat vague phrases like "the game", "the game and the world", "the story", "the lore", "here", or "at this point" as referring to that current game. Only switch to another game if the player clearly names another game in the current question.
2. Use the vision result as current-state context: current objective, location, menu text, visible options, map/journal clues, enemies, items, NPCs, or progression.
3. Use the online snippets as research context. Analyze them; do not quote or list search-result titles.
4. Use recent conversation history to resolve follow-ups like "it", "that", "who wrote it", "what about that", or "tell me more".
5. Final answer must synthesize all useful context: current game + current vision result + recent previous screen analyses + web snippets + history.
6. For recommendations/advice, suggest concrete names/options when available and briefly explain why. If exact availability depends on progression and you cannot verify it, say so briefly.
7. For practical/how-to questions, extract the actual steps from the web snippets and screen context. Answer with what to try, not with page titles.
8. If the vision result is unclear, still answer from the autodetected/typed game plus web/history.
9. Avoid spoilers unless the player asks for story details.

Keep it concise, natural, and useful. No markdown unless a short list makes the answer clearer.
""".strip()

def answer_with_one_general_prompt(question: str, gen: int) -> str:
    global runtime_game_context

    # Exact greetings/mic tests still stay local. Anything with a real question goes through full pipeline.
    if is_simple_chat_question(question):
        return answer_simple_chat(question)

    if looks_like_repeated_gibberish(question):
        return "I heard noise or repeated filler, not a clear question."

    history_direct = direct_answer_from_history(question)
    if history_direct:
        print("[History] answered directly from recent conversation")
        return history_direct

    extracted_game = extract_game_from_question_raw(question)
    if extracted_game and is_game_context_statement(question):
        runtime_game_context = extracted_game

    game = get_effective_game_for_question(question)
    print(f"[Context] current/target game: {game}")

    history = format_history_for_prompt()

    # Always capture and analyze screenshot for current-state context.
    image_b64 = ""
    screen_context = ""
    image_path, _age = get_capture_for_generation(gen, wait_timeout=4.0)
    if image_path:
        try:
            small = prepare_image_for_ollama(image_path)
            image_b64 = image_to_base64(small)
        except Exception as e:
            print(f"[ImagePrep error] {e}")

    if image_b64:
        screen_context = summarize_screen_context_for_question(question, image_b64, game)
        if screen_context:
            remember_screen_context(screen_context, image_path)
            print(f"[VisionResult] {screen_context}")
        else:
            print("[VisionResult] unclear/unusable")
    else:
        print("[VisionResult] no screenshot available")

    # Always run web for real questions. If disabled, web_search_snippets returns empty.
    print(f"[Web] game target: {game}")
    queries = build_search_queries(question, screen_context)
    print(f"[Web] searching {len(queries)} focused queries for: {question}")
    web = web_search_snippets(queries)

    screen_history = format_screen_context_history()
    prompt = build_one_general_prompt(question, game, history, web, screen_context, screen_history)

    message = {"role": "user", "content": prompt}

    answer = ollama_chat_safe(
        model=TEXT_MODEL,
        messages=[message],
        options={
            "num_ctx": ONE_PROMPT_NUM_CTX,
            "num_predict": ONE_PROMPT_NUM_PREDICT,
            "temperature": TEXT_TEMPERATURE,
            "top_k": TEXT_TOP_K,
        },
        label="TextFinal",
    )

    if answer and not looks_like_prompt_echo(answer) and not is_bad_generation_output(answer):
        return answer

    print(f"[TextFinal] unusable answer, falling back: {answer!r}")

    direct = direct_fact_from_snippets(question, web)
    if direct:
        print("[WebFacts] answered directly from snippets")
        return direct

    fallback = synthesized_fallback_answer(question, web, game, bool(screen_context or image_b64))
    if fallback:
        print("[SmartFallback] synthesized answer from context/snippets")
        return fallback

    if GAME_HINT and looks_like_game_name_question(question):
        return f"You are playing {GAME_HINT}."

    return "I could not answer that reliably from the available context."


# ---------------- ANSWERING ----------------

def answer_question(question: str, gen: int) -> str:
    answer_in_progress.set()
    try:
        answer = clamp_answer(answer_with_one_general_prompt(question, gen))
        remember_turn(question, answer)
        return answer
    finally:
        answer_in_progress.clear()


# ---------------- TTS ----------------

def stop_speaking():
    global tts_processes
    with tts_processes_lock:
        procs = list(tts_processes)
        tts_processes = []

    for p in procs:
        try:
            if p and p.poll() is None:
                p.terminate()
        except Exception:
            pass

    time.sleep(0.03)
    for p in procs:
        try:
            if p and p.poll() is None:
                p.kill()
        except Exception:
            pass

def speak(text: str):
    if not TTS_ENABLED:
        return

    text = clamp_answer(text)
    if not text:
        return

    piper_cmd = shutil.which("piper") or shutil.which("piper-tts")
    player_cmd = shutil.which("aplay")

    if not piper_cmd:
        print("[TTS error] Piper not found.")
        return
    if not player_cmd:
        print("[TTS error] aplay not found. Install: sudo pacman -S alsa-utils")
        return

    def _run():
        with speaking_lock:
            try:
                piper = subprocess.Popen(
                    [piper_cmd, "--model", VOICE_MODEL, "--output-raw"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )

                aplay = subprocess.Popen(
                    [player_cmd, "-r", "22050", "-f", "S16_LE", "-t", "raw", "-"],
                    stdin=piper.stdout,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

                with tts_processes_lock:
                    tts_processes.clear()
                    tts_processes.extend([piper, aplay])

                piper.stdin.write(text)
                piper.stdin.close()
                piper.wait()
                aplay.wait()

                with tts_processes_lock:
                    tts_processes.clear()

            except Exception as e:
                print(f"[TTS error] {e}")

    threading.Thread(target=_run, daemon=True).start()



def transcribe_audio(model, wav_path: str) -> tuple[str, dict]:
    """
    Better Whisper wrapper:
    - English language hint
    - optional Silero VAD inside faster-whisper
    - beam search for better accuracy
    - confidence stats for rejecting garbage/noise
    """
    kwargs = {
        "beam_size": WHISPER_BEAM_SIZE,
        "vad_filter": WHISPER_VAD_FILTER,
        "condition_on_previous_text": False,
        "temperature": 0.0,
        "language": WHISPER_LANGUAGE or None,
        "initial_prompt": WHISPER_INITIAL_PROMPT or None,
    }

    if WHISPER_VAD_FILTER:
        kwargs["vad_parameters"] = {
            "min_silence_duration_ms": 450,
            "speech_pad_ms": 250,
        }

    segments, info = model.transcribe(wav_path, **kwargs)
    segs = list(segments)
    text = " ".join(seg.text.strip() for seg in segs).strip()

    avg_logprob = None
    if segs:
        vals = [getattr(seg, "avg_logprob", None) for seg in segs]
        vals = [v for v in vals if v is not None]
        if vals:
            avg_logprob = sum(vals) / len(vals)

    no_speech_prob = None
    if segs:
        vals = [getattr(seg, "no_speech_prob", None) for seg in segs]
        vals = [v for v in vals if v is not None]
        if vals:
            no_speech_prob = sum(vals) / len(vals)

    meta = {
        "segments": len(segs),
        "avg_logprob": avg_logprob,
        "no_speech_prob": no_speech_prob,
    }
    return text, meta

def looks_like_bad_transcript(text: str, meta: dict) -> bool:
    """
    Reject common noise hallucinations from Whisper.
    Conservative: do not reject normal short commands like 'hello' or 'stop'.
    """
    t = (text or "").strip()
    if not t:
        return True

    words = re.findall(r"[A-Za-z']+", t)
    if len(words) == 0:
        return True

    no_speech = meta.get("no_speech_prob")
    avg_logprob = meta.get("avg_logprob")

    if no_speech is not None and no_speech >= WHISPER_NO_SPEECH_THRESHOLD and len(words) <= 5:
        return True

    if avg_logprob is not None and avg_logprob < WHISPER_MIN_AVG_LOGPROB:
        return True

    # Common hallucinations/garbage from very short noise snippets.
    garbage_phrases = (
        "thank you for watching",
        "thanks for watching",
        "subscribe",
        "like and subscribe",
        "you",
    )
    low = t.lower().strip(" .!?")
    if low in garbage_phrases:
        return True

    if looks_like_repeated_gibberish(t):
        return True

    return False

# ---------------- AUDIO ----------------

def save_wav(path: str, audio: np.ndarray):
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())

def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio))))

def list_input_devices():
    try:
        default_in, default_out = sd.default.device
        print("[Mic] Default input device:", default_in)
        print("[Mic] Default output device:", default_out)
        print("[Mic] Available input devices:")
        for i, dev in enumerate(sd.query_devices()):
            if dev.get("max_input_channels", 0) > 0:
                default_mark = " <default>" if i == default_in else ""
                print(f"{i}: {dev.get('name')}, inputs={dev.get('max_input_channels')}, default_sr={dev.get('default_samplerate'):.0f}{default_mark}")
    except Exception as e:
        print(f"[Mic] failed to list devices: {e}")

def mic_listener():
    if MIC_LIST_DEVICES:
        list_input_devices()

    frame_ms = 30
    frame_samples = int(SAMPLE_RATE * frame_ms / 1000)
    silence_frames_needed = int(SILENCE_END_SECONDS * 1000 / frame_ms)
    min_speech_frames = int(MIN_SPEECH_SECONDS * 1000 / frame_ms)
    analysis_start_frames = max(1, int(ANALYSIS_START_AFTER_SECONDS * 1000 / frame_ms))
    trigger_frames_needed = max(1, int(SPEECH_TRIGGER_SECONDS * 1000 / frame_ms))
    pre_roll_frames = max(1, int(PRE_ROLL_SECONDS * 1000 / frame_ms))

    while not stop_event.is_set():
        try:
            stream = sd.InputStream(
                device=INPUT_DEVICE,
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="float32",
                blocksize=frame_samples,
            )

            print(f"[Mic] InputStream created on device: {INPUT_DEVICE_RAW}")
            print("[Mic] continuous listening enabled")

            with stream:
                recording = []
                pre_roll = []
                speech_frames = 0
                trigger_frames = 0
                silence_frames = 0
                is_recording = False
                current_recording_gen = 0
                analysis_started = False
                last_level_print = 0.0

                while not stop_event.is_set():
                    data, _ = stream.read(frame_samples)
                    mono = data[:, 0]
                    level = rms(mono)

                    if MIC_DEBUG:
                        now = time.time()
                        if now - last_level_print >= 0.5:
                            last_level_print = now
                            state = "recording" if is_recording else "idle"
                            print(
                                f"[Mic] level={level:.5f} start={VOICE_START_THRESHOLD:.5f} "
                                f"continue={VOICE_CONTINUE_THRESHOLD:.5f} state={state}"
                            )

                    if not is_recording:
                        pre_roll.append(mono.copy())
                        if len(pre_roll) > pre_roll_frames:
                            pre_roll.pop(0)

                        if level >= VOICE_START_THRESHOLD:
                            trigger_frames += 1
                        else:
                            trigger_frames = 0

                        # Start only after sustained speech, not a one-frame pop/click.
                        if trigger_frames >= trigger_frames_needed:
                            is_recording = True
                            recording = list(pre_roll)
                            speech_frames = trigger_frames
                            silence_frames = 0
                            analysis_started = False
                            print("[Mic] speech started")
                            stop_speaking()
                            current_recording_gen = next_generation()

                        continue

                    # Already recording: use a lower continue threshold so quiet syllables are not chopped.
                    is_voice_frame = level >= VOICE_CONTINUE_THRESHOLD
                    recording.append(mono.copy())

                    if is_voice_frame:
                        speech_frames += 1
                        silence_frames = 0

                        # Start one screenshot capture after sustained speech.
                        # This avoids burning screenshots on false starts.
                        if not analysis_started and speech_frames >= analysis_start_frames:
                            analysis_started = True
                            start_capture_for_generation(current_recording_gen)
                    else:
                        silence_frames += 1

                    if silence_frames >= silence_frames_needed:
                        duration = len(recording) * frame_ms / 1000.0
                        if speech_frames >= min_speech_frames:
                            audio = np.concatenate(recording)
                            gen = current_recording_gen or next_generation()
                            if not analysis_started:
                                analysis_started = True
                                start_capture_for_generation(gen)
                            print(f"[Mic] speech captured ({duration:.2f}s)")
                            speech_queue.put((gen, audio))
                        else:
                            print("[Mic] ignored too-short speech")

                        is_recording = False
                        recording = []
                        pre_roll = []
                        speech_frames = 0
                        trigger_frames = 0
                        silence_frames = 0
                        analysis_started = False

        except Exception as e:
            print(f"[Mic error] {e}; retrying in 1s")
            time.sleep(1)

def speech_loop():
    print(f"[Whisper] loading {WHISPER_SIZE} on {WHISPER_DEVICE} / {WHISPER_COMPUTE_TYPE}...")
    try:
        model = WhisperModel(WHISPER_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
    except Exception as e:
        print(f"[Whisper] load failed: {e}")
        print("[Whisper] falling back to CPU / int8...")
        model = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")

    while not stop_event.is_set():
        gen, audio = speech_queue.get()

        try:
            total_start = time.time()

            wav_path = str(Path(tempfile.gettempdir()) / "gaming_agent_question.wav")
            save_wav(wav_path, audio)

            t0 = time.time()
            text, meta = transcribe_audio(model, wav_path)
            confidence_bits = []
            if meta.get("avg_logprob") is not None:
                confidence_bits.append(f"avg_logprob={meta['avg_logprob']:.2f}")
            if meta.get("no_speech_prob") is not None:
                confidence_bits.append(f"no_speech={meta['no_speech_prob']:.2f}")
            extra = " " + " ".join(confidence_bits) if confidence_bits else ""
            print(f"[Whisper] {time.time() - t0:.2f}s{extra}")

            if looks_like_bad_transcript(text, meta):
                print(f"[Whisper] ignored low-confidence/noise transcript: {text!r}")
                continue

            print(f"[You] {text}")
            print("[Agent] answering...")

            answer = answer_question(text, gen)

            if gen != current_generation():
                print("[Agent] discarded stale answer because a newer question started")
                continue

            print(f"[Agent] {answer}")
            print(f"[Total answer latency] {time.time() - total_start:.2f}s")
            speak(answer)

        except Exception as e:
            print(f"[Speech loop error] {e}")


# ---------------- WARMUP ----------------

def warm_image_model():
    if not WARM_IMAGE_MODEL:
        return

    try:
        out = Path(tempfile.gettempdir()) / "gaming_agent_warmup.jpg"
        img = Image.new("RGB", (IMAGE_MAX_WIDTH, IMAGE_MAX_HEIGHT), color=(0, 0, 0))
        img.save(out, "JPEG", quality=IMAGE_JPEG_QUALITY)
        img64 = image_to_base64(str(out))
        print("[Warmup] warming image path...")
        _ = ollama_chat_safe(
            model=VISION_MODEL,
            messages=[{"role": "user", "content": "Reply OK.", "images": [img64]}],
            options={"num_ctx": 256, "num_predict": 2, "temperature": VISION_TEMPERATURE, "top_k": VISION_TOP_K},
            label="WarmupVision",
        )
    except Exception as e:
        print(f"[Warmup error] {e}")




# ---------------- STEAM GAME DETECTION ----------------

def steam_library_roots() -> list[Path]:
    """
    Find Steam library roots. Handles the default Linux Steam path and libraryfolders.vdf.
    """
    roots = []
    candidates = [
        Path.home() / ".local/share/Steam",
        Path.home() / ".steam/steam",
        Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam",
    ]

    for root in candidates:
        if (root / "steamapps").exists() and root not in roots:
            roots.append(root)

    for root in list(roots):
        vdf = root / "steamapps/libraryfolders.vdf"
        if not vdf.exists():
            continue
        try:
            data = vdf.read_text(errors="ignore")
            for m in re.finditer(r'"path"\s+"([^"]+)"', data):
                p = Path(m.group(1).replace("\\\\", "/"))
                if (p / "steamapps").exists() and p not in roots:
                    roots.append(p)
        except Exception:
            pass

    return roots

def steam_app_manifest_names() -> dict[str, str]:
    """
    Map Steam appid -> installed app name using appmanifest_*.acf.
    """
    out = {}
    for root in steam_library_roots():
        steamapps = root / "steamapps"
        for manifest in steamapps.glob("appmanifest_*.acf"):
            try:
                txt = manifest.read_text(errors="ignore")
                appid_m = re.search(r'"appid"\s+"?(\d+)"?', txt)
                name_m = re.search(r'"name"\s+"([^"]+)"', txt)
                if appid_m and name_m:
                    out[appid_m.group(1)] = name_m.group(1)
            except Exception:
                pass
    return out

def running_steam_appids() -> list[str]:
    """
    Scan /proc for running Steam games.
    Steam-launched games commonly have SteamAppId/SteamGameId in their environment.
    """
    found = []
    proc = Path("/proc")

    for p in proc.iterdir():
        if not p.name.isdigit():
            continue

        try:
            cmdline = (p / "cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "ignore").lower()
        except Exception:
            cmdline = ""

        if "steamwebhelper" in cmdline or "steam-runtime" in cmdline:
            continue

        try:
            env_raw = (p / "environ").read_bytes().decode("utf-8", "ignore")
        except Exception:
            continue

        env = {}
        for part in env_raw.split("\x00"):
            if "=" in part:
                k, v = part.split("=", 1)
                env[k] = v

        appid = env.get("SteamAppId") or env.get("SteamGameId") or env.get("steam_appid")
        if appid and appid.isdigit() and appid not in ("0", "753"):
            if appid not in found:
                found.append(appid)

    return found

def detect_running_steam_game() -> tuple[str, str]:
    """
    Return (game_name, appid), or ("", "") if not detected.
    """
    if not AUTO_DETECT_STEAM_GAME:
        return "", ""

    appids = running_steam_appids()
    if not appids:
        return "", ""

    names = steam_app_manifest_names()

    for appid in appids:
        name = names.get(appid, "").strip()
        if name:
            return name, appid

    return f"Steam app {appids[0]}", appids[0]

def prompt_for_game_name():
    """
    Ask the player once at startup what game they are playing.
    Steam autodetection is used as a suggested default when available.
    """
    global GAME_HINT, runtime_game_context

    detected_name, detected_appid = detect_running_steam_game()
    if detected_name:
        print(f"[SteamDetect] detected running game: {detected_name} ({detected_appid})")

    if not PROMPT_GAME_NAME:
        if not GAME_HINT and detected_name:
            GAME_HINT = detected_name
            runtime_game_context = detected_name
        return

    current = GAME_HINT.strip() or detected_name.strip()
    try:
        if current:
            typed = input(f"Game name [{current}]: ").strip()
            if typed:
                GAME_HINT = typed
                runtime_game_context = typed
            else:
                GAME_HINT = current
                runtime_game_context = current
        else:
            typed = input("Game name, optional: ").strip()
            if typed:
                GAME_HINT = typed
                runtime_game_context = typed
    except EOFError:
        if current:
            GAME_HINT = current
            runtime_game_context = current
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"[GamePrompt warning] {e}")

# ---------------- MAIN ----------------

def run_thread(name: str, target):
    def wrapper():
        try:
            print(f"[{name}] starting")
            target()
        except Exception as e:
            print(f"[{name}] crashed: {repr(e)}")

    thread = threading.Thread(target=wrapper, daemon=True)
    thread.start()
    return thread

def main():
    prompt_for_game_name()
    print("Gaming agent starting: SPEECH-TRIGGERED SCREENSHOT MODE")
    print("Vision model:", VISION_MODEL)
    print("Vision temperature:", VISION_TEMPERATURE)
    print("Vision model mode:", "moondream" if is_moondream_model() else "general")
    print("Text model:", TEXT_MODEL)
    print("Text temperature:", TEXT_TEMPERATURE)
    print("One-prompt model:", ONE_PROMPT_MODEL)
    print("Game hint/current game:", GAME_HINT or runtime_game_context or "none")
    print("Prompt game name on start:", "yes" if PROMPT_GAME_NAME else "no")
    print("Web search:", "yes" if WEB_SEARCH_ENABLED else "no")
    print("Web max queries/results:", WEB_MAX_QUERIES, WEB_MAX_RESULTS)
    print("Steam autodetect:", "yes" if AUTO_DETECT_STEAM_GAME else "no")
    print("Background vision:", "yes" if BACKGROUND_VISION else "no", f"(every {SCREENSHOT_INTERVAL}s, last {SCREEN_CONTEXT_CACHE_SIZE} analyses)")
    print("Speech-start behavior: screenshot + vision context for every real question")
    print("Speech-triggered screenshot delay:", ANALYSIS_START_AFTER_SECONDS, "s")
    print("Screen analysis wait timeout:", SCREEN_ANALYSIS_WAIT_TIMEOUT, "s")
    print("Screenshot tool:", SCREENSHOT_TOOL)
    print("Save full screenshots as JPG:", "yes" if SAVE_SCREENSHOT_JPG else "no")
    print(f"Image size: {IMAGE_MAX_WIDTH}x{IMAGE_MAX_HEIGHT}, q={IMAGE_JPEG_QUALITY}")
    print(f"Whisper: {WHISPER_SIZE} on {WHISPER_DEVICE}/{WHISPER_COMPUTE_TYPE}, beam={WHISPER_BEAM_SIZE}, vad={WHISPER_VAD_FILTER}")
    print("Mic thresholds:", "start", VOICE_START_THRESHOLD, "continue", VOICE_CONTINUE_THRESHOLD)
    print("Sustained speech trigger:", SPEECH_TRIGGER_SECONDS, "s")
    print("Mic debug:", "yes" if MIC_DEBUG else "no")
    print("Conversation history turns:", HISTORY_TURNS)
    print("Auto-ack game corrections:", "yes" if AUTO_ACK_GAME_CORRECTIONS else "no")
    print("Local mic-test/greeting handling: yes")
    print("Screenshot/vision mode: always extract screen context")
    print("Screen context cache size:", SCREEN_CONTEXT_CACHE_SIZE)
    print("Clean /tmp agent images on start:", "yes" if CLEAN_TMP_IMAGES_ON_START else "no")
    print("Press Ctrl+C to stop.")

    cleanup_agent_tmp_images()
    warm_image_model()

    if BACKGROUND_VISION:
        run_thread("VisionCache", vision_cache_loop)

    run_thread("Mic", mic_listener)
    run_thread("Speech", speech_loop)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Agent] stopping...")
        stop_event.set()
        stop_speaking()
        time.sleep(0.5)

if __name__ == "__main__":
    main()
