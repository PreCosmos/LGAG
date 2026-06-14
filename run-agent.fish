#!/usr/bin/env fish

# Cached ultrafast gaming agent for Garuda/KDE/Fish + AMD RX 7900 XT ROCm

set AGENT_DIR (dirname (status --current-filename))
cd $AGENT_DIR

if test -f ".venv/bin/activate.fish"
    source .venv/bin/activate.fish
else
    echo "[Launch error] Missing .venv/bin/activate.fish"
    echo "Create it with: python -m venv .venv"
    exit 1
end

# AMD/ROCm visibility
set -x ROCR_VISIBLE_DEVICES 0
set -x HIP_VISIBLE_DEVICES 0

# Ollama performance/stability
set -x OLLAMA_KEEP_ALIVE 2m
set -x OLLAMA_FLASH_ATTENTION 1
set -x OLLAMA_HOST 127.0.0.1:11434

# Models
# The one general prompt receives text + screenshot, so this should be multimodal.
set -x GAG_VISION_MODEL qwen2.5vl:3b
set -x GAG_ONE_PROMPT_MODEL qwen2.5vl:3b
set -x GAG_VISION_TEMPERATURE 0.1
set -x GAG_VISION_TOP_K 20

# Text fallback only if no screenshot is available.
set -x GAG_TEXT_MODEL llama3.2:3b
# Alternative text models:
# set -x GAG_TEXT_MODEL qwen3:4b
# set -x GAG_TEXT_MODEL qwen3:8b
# Stronger but much slower:
# set -x GAG_TEXT_MODEL qwen3:8b
# More reliable text synthesis fallback if Qwen3 returns empty answers:
# set -x GAG_TEXT_MODEL llama3.2:3b
# Stronger but slower:
# set -x GAG_TEXT_MODEL qwen3:4b
set -x GAG_TEXT_TEMPERATURE 0.35
set -x GAG_TEXT_TOP_K 40
set -x GAG_ONE_PROMPT_NUM_CTX 4096
set -x GAG_ONE_PROMPT_NUM_PREDICT 220

# Ask for the game name at startup. Leave GAG_GAME_HINT blank here.
set -x GAG_GAME_HINT ""
set -x GAG_PROMPT_GAME_NAME 1
set -x GAG_AUTO_DETECT_STEAM_GAME 1

# Cached vision mode.
set -x GAG_SCREEN_CONTEXT_CACHE_SIZE 10
set -x GAG_BACKGROUND_VISION 1
set -x GAG_SCREENSHOT_INTERVAL 5
set -x GAG_MAX_CACHE_AGE 60
set -x GAG_SCREEN_ANALYSIS_WAIT_TIMEOUT 30
set -x GAG_ANALYSIS_START_AFTER_SECONDS 0.45
set -x GAG_FRESH_FALLBACK_IF_CACHE_EMPTY 0

# Live screen speed/accuracy settings.
set -x GAG_SCREENSHOT_TOOL spectacle
set -x GAG_SAVE_SCREENSHOT_JPG 1
set -x GAG_SCREENSHOT_JPEG_QUALITY 80
set -x GAG_CLEAN_TMP_IMAGES_ON_START 1
set -x GAG_IMAGE_MAX_WIDTH 1280
set -x GAG_IMAGE_MAX_HEIGHT 720
set -x GAG_IMAGE_JPEG_QUALITY 65
set -x GAG_VISION_NUM_CTX 2048
set -x GAG_VISION_NUM_PREDICT 120
set -x GAG_WARM_IMAGE_MODEL 1

# General web/lore settings.
set -x GAG_WEB_SEARCH 1
set -x GAG_WEB_TIMEOUT 3
set -x GAG_WEB_MAX_RESULTS 2
set -x GAG_WEB_MAX_QUERIES 2
set -x GAG_WEB_CACHE_TTL 300
set -x GAG_TEXT_NUM_CTX 1024
set -x GAG_TEXT_NUM_PREDICT 140
set -x GAG_HISTORY_TURNS 6
set -x GAG_HISTORY_MAX_CHARS 1200
set -x GAG_AUTO_ACK_GAME_CORRECTIONS 1

# Voice input.
set -x GAG_WHISPER_SIZE medium
set -x GAG_WHISPER_DEVICE cpu
set -x GAG_WHISPER_COMPUTE_TYPE int8
set -x GAG_WHISPER_BEAM_SIZE 3
set -x GAG_WHISPER_LANGUAGE en
set -x GAG_WHISPER_VAD_FILTER 1
set -x GAG_WHISPER_NO_SPEECH_THRESHOLD 0.75
set -x GAG_WHISPER_MIN_AVG_LOGPROB -1.05
set -x GAG_WHISPER_INITIAL_PROMPT "Gaming assistant. Common phrases: what game am I playing, what should I do, what does this say, tell me about the lore, compare this game with another game, what games are similar."
set -x GAG_INPUT_DEVICE default
set -x GAG_VOICE_THRESHOLD 0.012
set -x GAG_VOICE_START_THRESHOLD 0.013
set -x GAG_VOICE_CONTINUE_THRESHOLD 0.008
set -x GAG_SPEECH_TRIGGER_SECONDS 0.18
set -x GAG_PRE_ROLL_SECONDS 0.30
set -x GAG_MIN_SPEECH_SECONDS 0.55
set -x GAG_SILENCE_END_SECONDS 0.75
set -x GAG_MIC_DEBUG 0
set -x GAG_MIC_LIST_DEVICES 0

# TTS.
set -x GAG_TTS_ENABLED 1
set -x GAG_MAX_VOICE_ADVICE_CHARS 650

if not command -q spectacle
    echo "[Launch error] spectacle not found."
    echo "Install with: sudo pacman -S spectacle"
    exit 1
end

if not command -q aplay
    echo "[Launch error] aplay not found."
    echo "Install with: sudo pacman -S alsa-utils"
    exit 1
end

python -c "import ollama, sounddevice, faster_whisper, PIL, numpy" >/dev/null 2>&1
if test $status -ne 0
    echo "[Launch error] Missing Python packages."
    echo "pip install ollama sounddevice faster-whisper pillow numpy"
    exit 1
end

python -c "import ddgs" >/dev/null 2>&1
if test $status -ne 0
    echo "[Launch warning] ddgs not installed; web answers will be unavailable."
    echo "Install with: pip install ddgs"
end

curl -s http://127.0.0.1:11434/api/tags >/dev/null 2>&1
if test $status -ne 0
    echo "[Ollama] Starting ollama serve..."
    ollama serve >/tmp/gaming-agent-ollama.log 2>&1 &
    sleep 2
end

echo "[Ollama] Ensuring models are installed..."
ollama pull $GAG_VISION_MODEL
ollama pull $GAG_TEXT_MODEL

# Text model is not warmed on purpose; keeping it unloaded helps vision latency.

echo "[Ollama] Current loaded/running models:"
ollama ps

echo "[Agent] Starting Linux Gaming Agent! Please remain at your seats..."
python gag.py
