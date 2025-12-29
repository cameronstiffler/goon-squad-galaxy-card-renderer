import json
import os
import sys
import math
import textwrap
import re
from PIL import Image, ImageDraw, ImageFont
import openai
import requests
import random
import argparse
import base64
import io
try:
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore
    HAS_GOOGLE_GENAI = True
except ImportError:
    genai = None  # type: ignore
    types = None  # type: ignore
    HAS_GOOGLE_GENAI = False

# --- ENV LOADING ---
def load_env(env_path=".env"):
    """Lightweight .env loader so the script can run without shell-exported vars."""
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, 'r') as env_file:
            for line in env_file:
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, val = stripped.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                # Do not override values that are already set in the environment.
                if key and key not in os.environ:
                    os.environ[key] = val
    except Exception as e:
        print(f"[!] WARNING: Failed to load .env file: {e}")

load_env()

# Track key presence for fallbacks
HAS_OPENAI_KEY = bool(os.getenv("OPENAI_API_KEY"))
HAS_GEMINI_KEY = bool(os.getenv("GEMINI_API_KEY"))
VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", os.getenv("GOOGLE_CLOUD_LOCATION", "global"))
HAS_VERTEX_CONFIG = bool(VERTEX_PROJECT_ID)

# --- PROVIDER SELECTION ---
def detect_provider():
    """Determine which API provider to use based on available credentials or explicit override."""
    explicit = os.getenv("MODEL_PROVIDER", "").lower()
    if explicit in {"openai", "gemini", "vertex"}:
        return explicit
    if HAS_OPENAI_KEY:
        return "openai"
    if HAS_GEMINI_KEY:
        return "gemini"
    if HAS_VERTEX_CONFIG:
        return "vertex"
    return "openai"  # default to OpenAI; will fail loudly if no key is present

MODEL_PROVIDER = detect_provider()
USING_OPENAI = MODEL_PROVIDER == "openai"
USING_GEMINI = MODEL_PROVIDER == "gemini"
USING_VERTEX = MODEL_PROVIDER == "vertex"

# --- MODEL PATH HELPERS ---
def ensure_model_path(model_name: str) -> str:
    """Ensure Gemini model names include the 'models/' prefix."""
    return model_name if model_name.startswith("models/") else f"models/{model_name}"

# Optional Gemini client setup (only used if GEMINI_API_KEY is present/selected)
google_genai = None
ImageGenerationModel = None
if os.getenv("GEMINI_API_KEY"):
    try:
        import google.generativeai as google_genai  # type: ignore
        google_genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        ImageGenerationModel = getattr(google_genai, "ImageGenerationModel", None)
    except ImportError:
        print("[!] WARNING: google-generativeai not installed; Gemini provider will not work until it's available.")
    except Exception as e:
        print(f"[!] WARNING: Failed to initialize Gemini client: {e}")

# === CONFIGURATION ===
SOURCE_ICONS = "all.png"          # Master Icon Sheet (Left side traits)
SOURCE_ABILITY_ICONS = "ability icons.png" # Ability Text Icons
OUTPUT_DIR = "finished_cards"
GOON_JSON_MODEL = os.getenv("GOON_JSON_MODEL", "gpt-4-turbo")
ART_MODEL = os.getenv("ART_MODEL", "dall-e-3")
GEMINI_JSON_MODEL = os.getenv("GEMINI_JSON_MODEL", "gemini-1.5-pro")
GEMINI_ART_MODEL = os.getenv("GEMINI_ART_MODEL", "imagen-3.0-generate-001")
VERTEX_JSON_MODEL = os.getenv("VERTEX_JSON_MODEL", "gemini-1.5-pro")
VERTEX_ART_MODEL = os.getenv("VERTEX_ART_MODEL", "gemini-3-pro-image-preview")

# Deck-specific locations for JSON inputs, art, and outputs
DECK_CONFIG = {
    "pcu": {
        "json": os.path.join("deck_data", "pcu", "pcu_deck_strict.json"),
        "art": os.path.join("art", "pcu"),
        "output": os.path.join(OUTPUT_DIR, "pcu"),
    },
    "narc": {
        "json": os.path.join("deck_data", "narc", "narc_deck_strict.json"),
        "art": os.path.join("art", "narc"),
        "output": os.path.join(OUTPUT_DIR, "narc"),
    },
    "meat": {
        "json": os.path.join("deck_data", "meat", "meat_deck_strict.json"),
        "art": os.path.join("art", "meat"),
        "output": os.path.join(OUTPUT_DIR, "meat"),
    },
    "omni": {
        "json": os.path.join("deck_data", "omni", "omni_deck_strict.json"),
        "art": os.path.join("art", "omni"),
        "output": os.path.join(OUTPUT_DIR, "omni"),
    },
    "necro": {
        "json": os.path.join("deck_data", "necro", "necro_deck_strict.json"),
        "art": os.path.join("art", "necro"),
        "output": os.path.join(OUTPUT_DIR, "necro"),
    },
}

# --- GOOGLE GENAI HELPERS (Vertex/public Gemini via unified client) ---
_VERTEX_CLIENT = None

def _normalize_google_model_id(model: str, use_vertex: bool) -> str:
    """Vertex expects bare IDs; public Gemini expects models/<id>."""
    if use_vertex:
        return model[7:] if model.startswith("models/") else model
    return model if model.startswith("models/") else f"models/{model}"

def _ensure_vertex_credentials() -> str:
    """Set GOOGLE_APPLICATION_CREDENTIALS if missing, defaulting to ./vertex.json."""
    creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds:
        creds = os.path.abspath("vertex.json")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds
    if not os.path.isfile(creds):
        raise FileNotFoundError(f"GOOGLE_APPLICATION_CREDENTIALS points to '{creds}', which does not exist.")
    return creds

def _get_vertex_genai_client():
    """Create or return a cached Google GenAI client configured for Vertex."""
    global _VERTEX_CLIENT
    if _VERTEX_CLIENT:
        return _VERTEX_CLIENT
    if not HAS_GOOGLE_GENAI:
        raise RuntimeError("google-genai is not installed; Vertex provider is unavailable.")
    project_id = VERTEX_PROJECT_ID
    if not project_id:
        raise RuntimeError("VERTEX_PROJECT_ID (or GOOGLE_CLOUD_PROJECT) is not configured.")
    location = VERTEX_LOCATION or "global"
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", location)
    _ensure_vertex_credentials()
    _VERTEX_CLIENT = genai.Client(vertexai=True, project=project_id, location=location)
    return _VERTEX_CLIENT

# --- COORDINATES ---
COST_POS_X = 40           
COST_START_Y = 55
ICON_SPACING = 95         

NAME_END_X = 700  
NAME_Y = 25

TEXT_BOX_START_X = 75
TEXT_BOX_START_Y = 700
TEXT_WIDTH_CHARS = 42

def create_transparent_frame(faction):
    frame_path = os.path.join("elements", faction, "frame.png")
    print(f"   [+] Processing {frame_path}...")
    if not os.path.exists(frame_path):
        print(f"   [!] ERROR: Could not find {frame_path}.")
        sys.exit()

    img = Image.open(frame_path).convert("RGBA").resize((750, 1050))
    width, height = img.size
    datas = img.getdata()
    new_data = []

    for i, item in enumerate(datas):
        y = i // width
        x = i % width
        is_bright = (item[0] > 200 and item[1] > 200 and item[2] > 200)
        is_hole_zone = (y > 100 and y < 650 and x > 50)
        if is_bright and is_hole_zone:
            new_data.append((0, 0, 0, 0))
        else:
            new_data.append(item)
    img.putdata(new_data)
    return img

# --- HELPER: Draw dynamic circle ---
def create_circle_icon(size, color="#F5F5DC"):
    super_size = size * 4
    circle = Image.new('RGBA', (super_size, super_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(circle)
    draw.ellipse((0, 0, super_size-1, super_size-1), fill=color)
    return circle.resize((size, size), Image.Resampling.LANCZOS)

def get_assets(faction):
    print("--- STEP 1: PROCESSING ASSETS ---")
    if not os.path.exists(SOURCE_ICONS) or not os.path.exists(SOURCE_ABILITY_ICONS):
        print(f"[!] Critical Error: Missing icon files.")
        sys.exit()

    assets = {}
    assets['frame'] = create_transparent_frame(faction)

    # --- 1. LEFT SIDE ICONS ---
    icon_base_dir = "card_icons"
    faction_icon_dir = os.path.join(icon_base_dir, faction)
    def get_main_icon(filename):
        try:
            path = os.path.join(faction_icon_dir, filename)
            icon = Image.open(path).convert("RGBA")
            return icon.resize((85, 85), Image.Resampling.LANCZOS)
        except FileNotFoundError:
            print(f"   [!] WARNING: Icon not found: {path}")
            return create_placeholder_art((85,85), "ICON?")

    if faction == 'narc':
        assets['cost_wind'] = get_main_icon('wind_cost.png')
        assets['cost_meat'] = get_main_icon('meat_cost.png')
        assets['cost_gear'] = get_main_icon('gear_cost.png')
        assets['resist'] = get_main_icon('resist.png')
        assets['nounwind'] = get_main_icon('no_unwind.png')
        assets['trait_mechanical'] = get_main_icon('mechanical.png')
        assets['trait_biological'] = get_main_icon('biological.png')
        assets['faction'] = get_main_icon('faction.png')
    else: # PCU
        assets['cost_wind'] = get_main_icon('wind_cost.png')
        assets['cost_meat'] = get_main_icon('meat_cost.png')
        assets['cost_gear'] = get_main_icon('gear_cost.png')
        assets['resist'] = get_main_icon('resist.png')
        assets['nounwind'] = get_main_icon('no_unwind.png')
        assets['trait_mechanical'] = get_main_icon('mechanical.png')
        assets['trait_biological'] = get_main_icon('biological.png')
        assets['faction'] = get_main_icon('faction.png')

    # Rank icons are also now in the card_icons folder
    assets['rank_sl'] = get_main_icon('rank_sl.png') if os.path.exists(os.path.join(faction_icon_dir, 'rank_sl.png')) else None
    assets['rank_sg'] = get_main_icon('rank_sg.png') if os.path.exists(os.path.join(faction_icon_dir, 'rank_sg.png')) else None
    assets['rank_t'] = get_main_icon('rank_t.png') if os.path.exists(os.path.join(faction_icon_dir, 'rank_t.png')) else None

    # --- 2. ABILITY ICONS ---
    abil_sheet = Image.open(SOURCE_ABILITY_ICONS).convert("RGBA")
    aw, ah = abil_sheet.size
    a_row_h = ah / 3
    icon_w = aw / 6 # The sheet has 6 columns
    
    def get_abil_icon(row, col_index):
        top = row * a_row_h
        bottom = top + a_row_h
        left = col_index * icon_w
        right = left + icon_w
        icon = abil_sheet.crop((left, top, right, bottom))
        bbox = icon.getbbox()
        if bbox: 
            icon = icon.crop(bbox)
        return icon

    # Resize icons after cropping to maintain aspect ratio unless specified otherwise
    assets['abil_meat'] = get_abil_icon(2, 2).resize((30, 30), Image.Resampling.LANCZOS)
    assets['abil_gear'] = get_abil_icon(2, 3).resize((30, 30), Image.Resampling.LANCZOS)
    
    # Special resize for passive icon
    passive_icon = get_abil_icon(2, 4)
    assets['abil_passive'] = passive_icon.resize((30, 15), Image.Resampling.LANCZOS)

    assets['abil_star'] = get_abil_icon(2, 5).resize((30, 30), Image.Resampling.LANCZOS)

    print("   [+] Assets loaded.")
    return assets

def get_font(candidates, size, default_font=ImageFont.load_default()):
    """Tries to load a font from a list of candidates."""
    for font_name in candidates:
        # First, try to load from a local 'fonts' directory
        try:
            return ImageFont.truetype(os.path.join("fonts", font_name), size)
        except IOError:
            # If that fails, try loading from the system
            try:
                return ImageFont.truetype(font_name, size)
            except IOError:
                continue
    return default_font

def create_placeholder_art(size=(650, 600), text="ART MISSING"):
    """Creates a placeholder image for missing artwork."""
    img = Image.new('RGBA', size, (50, 50, 50, 255))
    draw = ImageDraw.Draw(img)
    font = get_font(["Georgia", "Arial"], 40)
    draw.text((size[0]/2, size[1]/2), text, font=font, anchor="mm", fill=(200, 200, 200))
    return img

def load_art_style_prompt(faction):
    """Load the faction-specific art style prompt, with a fallback to the legacy root file."""
    style_paths = [
        os.path.join("goon_design_guide", faction, "art_style.json"),
        "art_style.json",
    ]
    last_error = None
    for path in style_paths:
        try:
            with open(path, 'r') as f:
                style_data = json.load(f)
            return style_data.get("art_style_description", "A character illustration.")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            last_error = (path, e)

    if last_error:
        path, err = last_error
        print(f"   [!] WARNING: Could not load art style file '{path}'. Using default prompt. Error: {err}")
    return "90s video game concept art, semi-realistic sticker style, bold comic-book inking. The image must be in full color and feature only a single character."

def load_art_prompt_data(faction):
    """Loads the art prompt template and options from a faction-specific guide."""
    guide_path = f"goon_design_guide/{faction}/goon_traits.json"
    try:
        with open(guide_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"   [!] WARNING: Could not load or parse art prompt guide at '{guide_path}'. Art generation may be generic. Error: {e}")
        return None

def load_style_image(faction):
    """Load an optional style reference image to send to Gemini."""
    base = f"goon_design_guide/{faction}/style_example"
    for ext, mime in ((".png", "image/png"), (".jpg", "image/jpeg"), (".jpeg", "image/jpeg")):
        path = base + ext
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    return mime, f.read()
            except Exception as e:
                print(f"   [!] WARNING: Failed to read style image '{path}': {e}")
    return None

def generate_art_prompt(goon, prompt_data, art_style_prompt):
    """Constructs a detailed prompt for DALL-E using a base style, character template, and goon-specific overrides."""
    if not prompt_data:
        print("     [!] WARNING: ai_art_prompt data not found in JSON. Skipping art generation.")
        return None

    template = prompt_data.get("template", "")
    options = prompt_data.get("options", {})
    quantities = prompt_data.get("option_trait_quantities", {})

    goon_name = goon.get('name', 'Unnamed Goon') if isinstance(goon, dict) else str(goon)
    description_traits = goon.get('description', {}) if isinstance(goon, dict) else {}

    if isinstance(template, list):
        template = "\n".join(template)
    elif not isinstance(template, str):
        template = str(template)

    description_aliases = {
        # Map template placeholders to possible description keys
        'armor': ['attire'],
        'head_wear': ['headgear', 'head gear', 'headwear'],
        'accessories': ['accessory'],
        'weapon': ['weaponry'],
        'facehair': ['face hair', 'facial hair', 'facial_hair', 'beard', 'beards'],
    }
    option_aliases = {
        # Handle small naming mismatches between template placeholders and options keys
        'perspective_angle': 'perspective angle',
        'facehair': 'face hair',
    }

    def normalize_description_value(value):
        """Convert description values into a clean string for prompt insertion."""
        if isinstance(value, list):
            cleaned = [str(item).strip() for item in value if str(item).strip()]
            return ", ".join(cleaned) if cleaned else "none"
        if isinstance(value, str):
            stripped = value.strip()
            return stripped if stripped else "none"
        if value is None:
            return None
        return str(value)

    def get_description_value(field):
        """Try to pull a field value from the goon's description (with aliases)."""
        if not isinstance(description_traits, dict):
            return None
        keys_to_check = [field] + description_aliases.get(field, [])
        for key in keys_to_check:
            if key in description_traits:
                return normalize_description_value(description_traits[key])
        return None

    def get_option_values(field):
        """Fetch the option list for a placeholder, accounting for alias keys."""
        if field in options and options[field]:
            return options[field]
        alt_key = option_aliases.get(field)
        if alt_key and alt_key in options and options[alt_key]:
            return options[alt_key]
        return None

    format_args = {'goon_name': goon_name}

    placeholder_fields = {match for match in re.findall(r"{([^}]+)}", template)}
    placeholder_fields.discard('goon_name')

    for field in placeholder_fields:
        description_value = get_description_value(field)
        if description_value not in (None, ""):
            format_args[field] = description_value
            continue

        option_values = get_option_values(field)
        if option_values:
            # Use the quantity specified in option_trait_quantities, default to 1
            quantity = max(1, quantities.get(field, 1))
            # Ensure we don't request more options than are available
            quantity = min(quantity, len(option_values))
            if quantity > 0:
                selected_options = random.sample(option_values, quantity)
                format_args[field] = ", ".join(selected_options)
                continue
        format_args[field] = f"default_{field}"

    character_description = template.format(**format_args)

    priority_note = "Use the above art style exactly. Do not introduce any new style cues; only apply the subject details below."
    return f"{art_style_prompt}\n\n{priority_note}\n\nSubject: {character_description}"

def _next_numbered_art_index(art_dir, prefix):
    """Find the next available numeric suffix for generated art files."""
    pattern = re.compile(rf"^{re.escape(prefix)}[_-]?(\d+)\.(?:jpe?g|png)$", re.IGNORECASE)
    max_idx = 0
    try:
        for fname in os.listdir(art_dir):
            match = pattern.match(fname)
            if match:
                max_idx = max(max_idx, int(match.group(1)))
    except FileNotFoundError:
        return 1
    return max_idx + 1

def generate_art_only_batch(faction, art_dir, count):
    """Generate standalone art portraits using the faction's trait/style guides."""
    print(f"\n--- ART-ONLY GENERATION ({faction.upper()}) ---")
    if count <= 0:
        print("   [!] No art generated because count was not positive.")
        return 0

    prompt_data = load_art_prompt_data(faction)
    if not prompt_data:
        print("   [!] Art prompt data missing; cannot generate art.")
        sys.exit(1)

    art_style_prompt = load_art_style_prompt(faction)
    style_image = load_style_image(faction)
    os.makedirs(art_dir, exist_ok=True)

    prefix = f"{faction}_art"
    start_idx = _next_numbered_art_index(art_dir, prefix)
    generated = 0

    for offset in range(count):
        idx = start_idx + offset
        goon_stub = {"name": f"{faction.upper()}_ART_{idx:03d}", "description": {}}
        prompt = generate_art_prompt(goon_stub, prompt_data, art_style_prompt)
        if not prompt:
            print("   [!] Skipping entry due to missing prompt.")
            continue
        filename = os.path.join(art_dir, f"{prefix}_{idx:03d}.jpg")
        if generate_and_save_art(prompt, filename, style_image=style_image):
            generated += 1

    print(f"   [+] Generated {generated}/{count} portraits to {art_dir}")
    return generated

def generate_and_save_art(prompt, save_path, style_image=None):
    """Generates art using the selected provider and saves it to the specified path."""
    print(f"     [+] Generating AI art for: {os.path.basename(save_path)}...")
    print(f"     [+] Using Prompt: {prompt}")
    if USING_VERTEX:
        return generate_and_save_art_vertex(prompt, save_path, style_image=style_image)
    if USING_GEMINI:
        return generate_and_save_art_gemini(prompt, save_path, style_image=style_image)
    return generate_and_save_art_openai(prompt, save_path)

def generate_and_save_art_openai(prompt, save_path):
    """Generates art using OpenAI's image endpoint."""
    try:
        client = openai.OpenAI() # This line reads the key from the environment variable
        response = client.images.generate(
            model=ART_MODEL,
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url
        
        # Download and save the image
        image_data = requests.get(image_url).content

        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'wb') as handler:
            handler.write(image_data)
        print(f"     [+] AI art saved to {save_path}")
        return True
    except Exception as e:
        print(f"     [!] AI art generation failed: {e}")
        return False

def _extract_gemini_inline_image(response):
    """Attempt to pull inline image bytes out of a Gemini response."""
    try:
        parts = getattr(response, "parts", None)
        if parts:
            for part in parts:
                inline = getattr(part, "inline_data", None)
                if inline and getattr(inline, "data", None):
                    data = inline.data
                    if isinstance(data, bytes):
                        return data
                    try:
                        return base64.b64decode(data)
                    except Exception:
                        return None
        if not response or not getattr(response, "candidates", None):
            return None
        for candidate in response.candidates:
            parts = getattr(candidate, "content", None)
            parts = getattr(parts, "parts", []) if parts else []
            for part in parts:
                inline = getattr(part, "inline_data", None)
                if inline and getattr(inline, "data", None):
                    data = inline.data
                    if isinstance(data, bytes):
                        return data
                    try:
                        return base64.b64decode(data)
                    except Exception:
                        return None
    except Exception:
        return None
    return None

def summarize_gemini_response(response):
    """Return a short string describing Gemini response parts for debugging."""
    summaries = []
    for idx, candidate in enumerate(getattr(response, "candidates", [])):
        reason = getattr(candidate, "finish_reason", None) or "-"
        part_types = []
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) if content else []:
            if getattr(part, "inline_data", None):
                part_types.append("inline_data")
            elif getattr(part, "text", None):
                part_types.append("text")
            else:
                part_types.append(type(part).__name__)
        summaries.append(f"cand{idx}: reason={reason}, parts={','.join(part_types) or 'none'}")
    return "; ".join(summaries) if summaries else "no candidates"

def _coerce_image_payload_to_bytes(payload):
    """Normalize various Gemini image payload shapes into raw bytes."""
    if payload is None:
        return None
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, str):
        try:
            return base64.b64decode(payload)
        except Exception:
            return None
    if isinstance(payload, Image.Image):
        buffer = io.BytesIO()
        payload.save(buffer, format="PNG")
        return buffer.getvalue()
    # Common attribute names on Gemini image objects
    for attr in ("image", "image_bytes", "_image_bytes", "bytes", "data"):
        val = getattr(payload, attr, None)
        if val:
            coerced = _coerce_image_payload_to_bytes(val)
            if coerced:
                return coerced
    return None

def _extract_data_uri_image(text_value):
    """Extract base64 image bytes from a data URI inside text."""
    if not isinstance(text_value, str):
        return None
    match = re.search(r"data:image/(png|jpeg);base64,([A-Za-z0-9+/=]+)", text_value)
    if not match:
        return None
    b64_data = match.group(2)
    try:
        return base64.b64decode(b64_data)
    except Exception:
        return None

def generate_and_save_art_gemini(prompt, save_path, style_image=None):
    """Generates art using Gemini (requires google-generativeai)."""
    if not google_genai:
        print("     [!] Gemini provider selected but google-generativeai is not installed.")
        return False
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        image_bytes = None
        response_summary = None
        model_name = ensure_model_path(GEMINI_ART_MODEL)

        # Prepare optional style image part
        style_part = None
        if style_image:
            mime_type, data_bytes = style_image
            style_part = {"mime_type": mime_type, "data": data_bytes}

        # 1) Try ImageGenerationModel for imagen-style models.
        if ImageGenerationModel and model_name.startswith("models/imagen"):
            try:
                model = ImageGenerationModel.from_pretrained(model_name)
                response = model.generate_images(prompt=prompt, image=style_part["data"] if style_part else None)
                response_summary = summarize_gemini_response(response)
                images = getattr(response, "images", None) or getattr(response, "generated_images", None)
                if images:
                    first_image = images[0]
                    image_bytes = _coerce_image_payload_to_bytes(first_image)
            except Exception as e:
                print(f"     [!] Gemini ImageGenerationModel call failed: {e}")

        # 2) Try GenerativeModel.generate_images if available (e.g., gemini-3-pro-image-preview).
        if image_bytes is None:
            model = google_genai.GenerativeModel(model_name)
            if hasattr(model, "generate_images"):
                try:
                    if style_part:
                        response = model.generate_images(
                            prompt=prompt,
                            images=[style_part],
                        )
                    else:
                        response = model.generate_images(prompt=prompt)
                    response_summary = summarize_gemini_response(response)
                    images = getattr(response, "generated_images", None) or getattr(response, "images", None)
                    if images:
                        image_bytes = _coerce_image_payload_to_bytes(images[0])
                except Exception as e:
                    print(f"     [!] Gemini generate_images failed: {e}")

        # 3) Fallback to generate_content and look for inline data/data URIs.
        if image_bytes is None:
            parts = []
            if style_part:
                parts.append(style_part)
                parts.append({"text": "Style reference image; match line quality, palette, and rendering style."})
            parts.append({"text": prompt})

            response = model.generate_content(parts, generation_config={"temperature": 0.9})
            response_summary = summarize_gemini_response(response)
            inline_bytes = _extract_gemini_inline_image(response)
            if inline_bytes:
                image_bytes = inline_bytes
            if image_bytes is None:
                generated_images = getattr(response, "generated_images", None) or getattr(response, "images", None)
                if generated_images:
                    image_bytes = _coerce_image_payload_to_bytes(generated_images[0])
            if image_bytes is None:
                # Check if any candidate text contains a data URI we can decode.
                for candidate in getattr(response, "candidates", []):
                    content = getattr(candidate, "content", None)
                    for part in getattr(content, "parts", []) if content else []:
                        text_val = getattr(part, "text", None)
                        if text_val:
                            maybe_bytes = _extract_data_uri_image(text_val)
                            if maybe_bytes:
                                image_bytes = maybe_bytes
                                break
                    if image_bytes:
                        break

        if not image_bytes:
            if response_summary:
                print(f"     [!] Gemini art generation did not return an image payload. Response: {response_summary}")
            else:
                print("     [!] Gemini art generation did not return an image payload.")
            return False

        # Some SDKs return a base64-encoded string; normalize to bytes first.
        if isinstance(image_bytes, str):
            try:
                image_bytes = base64.b64decode(image_bytes)
            except Exception as e:
                print(f"     [!] Gemini returned a string payload that could not be base64-decoded: {e}")
                return False

        # Validate and, if it fails, write a raw debug file for inspection.
        try:
            Image.open(io.BytesIO(image_bytes)).verify()
        except Exception as e:
            head = image_bytes[:32]
            print(f"     [!] Gemini returned data that is not a valid image: {e}")
            print(f"         Bytes len={len(image_bytes)}, head={head!r}")
            if response_summary:
                print(f"         Response: {response_summary}")
            debug_path = save_path + ".raw"
            try:
                with open(debug_path, "wb") as dbg:
                    dbg.write(image_bytes)
                print(f"         Raw payload saved to {debug_path} for inspection.")
            except Exception as write_err:
                print(f"         Failed to save raw payload: {write_err}")
            return False

        with open(save_path, 'wb') as handler:
            handler.write(image_bytes)
        print(f"     [+] AI art saved to {save_path}")
        return True
    except Exception as e:
        print(f"     [!] Gemini art generation failed: {e}")
        return False

def generate_and_save_art_vertex(prompt, save_path, style_image=None):
    """Generates art using Vertex AI via the unified google-genai client."""
    if not HAS_GOOGLE_GENAI:
        print("     [!] Vertex provider selected but google-genai is not installed.")
        return False
    if not VERTEX_PROJECT_ID:
        print("     [!] Vertex provider selected but VERTEX_PROJECT_ID/GOOGLE_CLOUD_PROJECT is not set.")
        return False
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        client = _get_vertex_genai_client()
        model_id = _normalize_google_model_id(VERTEX_ART_MODEL, use_vertex=True)

        parts = []
        if style_image:
            mime_type, data_bytes = style_image
            data_bytes = _coerce_image_payload_to_bytes(data_bytes)
            if data_bytes:
                encoded = base64.b64encode(data_bytes).decode("ascii")
                parts.append({"inline_data": {"mime_type": mime_type, "data": encoded}})
                parts.append({"text": "Style reference image; match line quality, palette, and rendering style."})
            else:
                print("     [-] Style image could not be encoded; continuing without it.")
        parts.append({"text": prompt})

        response = client.models.generate_content(
            model=model_id,
            contents=parts,
            config=types.GenerateContentConfig(
                temperature=0.9,
                response_modalities=["TEXT", "IMAGE"],
            ),
        )

        image_bytes = _extract_gemini_inline_image(response)
        if not image_bytes:
            generated_images = getattr(response, "generated_images", None) or getattr(response, "images", None)
            if generated_images:
                image_bytes = _coerce_image_payload_to_bytes(generated_images[0])
        if not image_bytes:
            debug_summary = summarize_gemini_response(response)
            print(f"     [!] Vertex art generation did not return an image payload. Response: {debug_summary}")
            return False

        if isinstance(image_bytes, str):
            image_bytes = base64.b64decode(image_bytes)

        Image.open(io.BytesIO(image_bytes)).verify()
        with open(save_path, "wb") as handler:
            handler.write(image_bytes)
        print(f"     [+] AI art saved to {save_path}")
        return True
    except Exception as e:
        print(f"     [!] Vertex AI art generation failed: {e}")
        return False

def create_grid_image(card_files, output_folder):
    """Arranges all generated cards into a single grid image."""
    if not card_files:
        print("\n--- No cards to generate grid. ---")
        return

    print("\n--- STEP 3: GENERATING GRID IMAGE ---")

    # Grid layout settings
    COLS = 10
    TARGET_GRID_WIDTH = 9000
    
    # Calculate dimensions
    card_width = TARGET_GRID_WIDTH // COLS
    # Maintain aspect ratio (original is 750x1050)
    card_height = int(card_width * (1050 / 750))
    
    num_cards = len(card_files)
    rows = math.ceil(num_cards / COLS)
    
    grid_width = COLS * card_width
    grid_height = rows * card_height

    # Create the grid canvas
    grid_image = Image.new("RGB", (grid_width, grid_height), "black")

    # Paste cards into the grid
    for i, card_file in enumerate(card_files):
        row = i // COLS
        col = i % COLS
        x_pos = col * card_width
        y_pos = row * card_height
        
        card_img = Image.open(card_file).resize((card_width, card_height), Image.Resampling.LANCZOS)
        grid_image.paste(card_img, (x_pos, y_pos))

    grid_path = os.path.join(output_folder, "grid.jpg")
    grid_image.save(grid_path, "JPEG", quality=95)
    print(f"   [+] Grid image saved to: {grid_path}")
    print(f"   [+] Dimensions: {grid_width}x{grid_height} pixels")
    print(f"   [+] Layout: {rows} rows, {COLS} columns")
    print(f"   [+] Total cards included: {num_cards}")

def normalize_goon_data(goon, faction, fix=False):
    """Ensures a goon's data is in a consistent, usable format."""
    if fix:
        goon.setdefault('name', 'Unnamed Goon')
        goon.setdefault('rank', 'BG')
        goon.setdefault('duplicates', 1)
        goon.setdefault('faction', faction)
        goon.setdefault('biological', False)
        goon.setdefault('mechanical', False)
        goon.setdefault('resist', False)
        goon.setdefault('no_unwind', False)
        goon.setdefault('deploy_requirements', [])
        goon.setdefault('abilities', [])
        goon.setdefault('portrait_art', [])
        
        for ability in goon.get('abilities', []):
            ability.setdefault('name', 'Unnamed Ability')
            ability.setdefault('cost', {'wind': 0, 'meat': 0, 'gear': 0})
            ability.setdefault('passive', False)
            ability.setdefault('must_use', False)
            ability.setdefault('text', "")

    def parse_cost_value(val):
        """Coerce cost values into ints when possible while preserving 'X'."""
        if isinstance(val, str):
            stripped = val.strip()
            if stripped.upper() == "X":
                return "X"
            match = re.search(r"-?\\d+", stripped)
            return int(match.group(0)) if match else 0
        if isinstance(val, (int, float)):
            return int(val)
        return 0

    # Normalize ability cost structures so rendering logic can rely on dict access
    for ability in goon.get('abilities', []):
        cost = ability.get('cost', {})
        if isinstance(cost, int) or isinstance(cost, str):
            cost = {'wind': cost, 'meat': 0, 'gear': 0}
        elif not isinstance(cost, dict):
            cost = {'wind': 0, 'meat': 0, 'gear': 0}

        cost.setdefault('wind', 0)
        cost.setdefault('meat', 0)
        cost.setdefault('gear', 0)
        cost['wind'] = parse_cost_value(cost.get('wind', 0))
        cost['meat'] = parse_cost_value(cost.get('meat', 0))
        cost['gear'] = parse_cost_value(cost.get('gear', 0))
        ability['cost'] = cost


    # Normalize portrait art entries into a list of non-empty strings
    portrait_art = goon.get('portrait_art', [])
    if isinstance(portrait_art, str):
        portrait_art = [portrait_art] if portrait_art.strip() else []
    elif isinstance(portrait_art, list):
        portrait_art = [p for p in portrait_art if isinstance(p, str) and p.strip()]
    else:
        portrait_art = []
    goon['portrait_art'] = portrait_art

    # --- Normalize deploy_cost ---
    deploy_cost = goon.get('deploy_cost', {})
    if isinstance(deploy_cost, int) or isinstance(deploy_cost, str):
        # If it's an int or string, assume it's a wind cost for legacy reasons.
        deploy_cost = {'wind': deploy_cost, 'meat': 0, 'gear': 0}
    
    # Ensure all cost types are present.
    deploy_cost.setdefault('wind', 0)
    deploy_cost.setdefault('meat', 0)
    deploy_cost.setdefault('gear', 0)
    deploy_cost['wind'] = parse_cost_value(deploy_cost.get('wind', 0))
    deploy_cost['meat'] = parse_cost_value(deploy_cost.get('meat', 0))
    deploy_cost['gear'] = parse_cost_value(deploy_cost.get('gear', 0))
    
    goon['deploy_cost'] = deploy_cost
    
    return goon


def validate_goon_schema(goon):
    """Validate that a goon dictionary has the required shape."""
    required_card_keys = [
        "name",
        "rank",
        "duplicates",
        "faction",
        "deploy_cost",
        "biological",
        "mechanical",
        "resist",
        "no_unwind",
        "deploy_requirements",
        "abilities",
        "portrait_art",
    ]
    required_ability_keys = ["name", "cost", "passive", "must_use", "text"]

    errors = []
    for key in required_card_keys:
        if key not in goon:
            errors.append(f"missing card field '{key}'")

    # -- Validate deploy_cost structure --
    deploy_cost = goon.get('deploy_cost', {})
    if not isinstance(deploy_cost, dict):
        errors.append("'deploy_cost' must be a dictionary.")
    else:
        for cost_type in ['wind', 'meat', 'gear']:
            if cost_type not in deploy_cost:
                errors.append(f"'deploy_cost' is missing '{cost_type}' key.")

    portrait_art = goon.get("portrait_art")
    if not isinstance(portrait_art, list):
        errors.append("portrait_art must be a list of filenames.")
    elif not all(isinstance(p, str) for p in portrait_art):
        errors.append("portrait_art must only contain strings.")

    abilities = goon.get("abilities", [])
    if not isinstance(abilities, list):
        errors.append("abilities should be a list")
        return errors

    for ability in abilities:
        for key in required_ability_keys:
            if key not in ability:
                errors.append(f"ability '{ability.get('name', '?')}' missing field '{key}'")

    return errors

def generate_goon_text_openai(prompt):
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=GOON_JSON_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
    )
    return response.choices[0].message.content

def generate_goon_text_gemini(prompt):
    if not google_genai:
        raise RuntimeError("Gemini provider selected but google-generativeai is not installed.")
    model = google_genai.GenerativeModel(ensure_model_path(GEMINI_JSON_MODEL))
    response = model.generate_content(prompt, generation_config={"temperature": 0.8})
    return response.text

def generate_goon_text_vertex(prompt):
    if not HAS_GOOGLE_GENAI:
        raise RuntimeError("Vertex provider selected but google-genai is not installed.")
    if not VERTEX_PROJECT_ID:
        raise RuntimeError("Vertex provider selected but VERTEX_PROJECT_ID/GOOGLE_CLOUD_PROJECT is not configured.")
    client = _get_vertex_genai_client()
    model_id = _normalize_google_model_id(VERTEX_JSON_MODEL, use_vertex=True)
    response = client.models.generate_content(
        model=model_id,
        contents=[{"text": prompt}],
        config=types.GenerateContentConfig(temperature=0.8, response_modalities=["TEXT"]),
    )
    return response.text

def generate_goon_text(prompt):
    if USING_VERTEX:
        return generate_goon_text_vertex(prompt)
    if USING_GEMINI:
        return generate_goon_text_gemini(prompt)
    return generate_goon_text_openai(prompt)

def generate_new_goon(faction, deck_json_path, goon_name=None):
    """Uses AI to generate a new goon and add it to the deck JSON file."""
    print(f"\n--- STEP 1: GENERATING NEW GOON FOR {faction.upper()} FACTION ---")
    
    # 1. Load the design guide
    guide_path = f"goon_design_guide/{faction}/creation_guide.json"
    try:
        with open(guide_path, 'r') as f:
            design_guide = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[!] ERROR: Could not load or parse the design guide at '{guide_path}'. Error: {e}")
        sys.exit()

    # 2. Construct the prompt for the AI
    prompt = f"""
You are a creative game designer for a card game called 'Goon Squad Galaxy'. Your task is to invent a new goon for the '{faction.upper()}' faction based on its official design guide.
"""
    if goon_name:
        prompt += f"\nThe goon's name must be '{goon_name}'."

    prompt += f"""
**Design Guide:**
```json
{json.dumps(design_guide, indent=2)}
```

**Instructions:**
1.  Internalize the design guide's high concept, tone, and visual style.
2.  Create a single, unique goon that fits perfectly within this faction.
3.  The output MUST be a single, valid JSON object representing the new goon. Do not include any explanatory text or markdown formatting around the JSON.
4.  The JSON object must conform to the structure of existing goons in the deck, including fields like "name", "rank", "deploy_cost", "abilities", etc.
5.  Provide at least one `portrait_art` filename (array of strings) ending in `.jpg`.
"""

    # 3. Call the AI to generate the goon JSON
    print("   [+] Prompting AI to generate new goon...")
    try:
        goon_json_string = generate_goon_text(prompt)

        # --- NEW: Clean the AI response to extract only the JSON object ---
        # Find the start and end of the JSON block
        start_index = goon_json_string.find('{')
        end_index = goon_json_string.rfind('}')
        if start_index != -1 and end_index != -1:
            clean_json_string = goon_json_string[start_index:end_index+1]
            new_goon = json.loads(clean_json_string)
        else: raise ValueError("No valid JSON object found in the AI response.")

        print(f"   [+] AI generated goon: {new_goon.get('name', 'Unnamed Goon')}")

        new_goon = normalize_goon_data(new_goon, faction, fix=True)

        # Validate schema before writing
        schema_errors = validate_goon_schema(new_goon)
        if schema_errors:
            print("[!] ERROR: Generated goon failed schema checks:")
            for err in schema_errors:
                print(f"      - {err}")
            sys.exit()
    except Exception as e:
        print(f"[!] ERROR: Failed to generate or parse AI response. Error: {e}")
        sys.exit()

    # 4. Add the new goon to the deck file
    try:
        with open(deck_json_path, 'r+') as f:
            deck_data = json.load(f)
            deck_data['goons'].append(new_goon)
            f.seek(0) # Rewind to the beginning of the file
            json.dump(deck_data, f, indent=2)
            f.truncate() # Remove any trailing data if the new file is shorter
        print(f"   [+] Successfully added '{new_goon.get('name')}' to {deck_json_path}")
        print("\n--- SUCCESS ---")
    except Exception as e:
        print(f"[!] ERROR: Failed to update the deck file '{deck_json_path}'. Error: {e}")
        sys.exit()

def generate_cards(json_file, art_dir, output_dir, faction, auto_generate_art=False, auto_extra_variations=0, create_grid=False, use_duplicates=False, fix=False):
    print("\n--- STEP 2: GENERATING CARDS ---")
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[!] ERROR: JSON file not found at '{json_file}'")
        sys.exit()
    except json.JSONDecodeError:
        print(f"[!] ERROR: Could not decode JSON from '{json_file}'. Please check for syntax errors.")
        sys.exit()
        
    assets = get_assets(faction)
    ai_prompt_data = load_art_prompt_data(faction)
    art_style_prompt = load_art_style_prompt(faction) # Load the master art style once
    style_image = load_style_image(faction)
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # --- FONTS (Loaded from JSON) ---
    def get_font_from_json(key_family, key_size, default_family, default_size):
        family = data.get(key_family, default_family)
        size = data.get(key_size, default_size)
        # Ensure family is a list for get_font function
        if not isinstance(family, list):
            family = [family]
        return get_font(family, size)

    font_header = get_font_from_json("card_name_font_family", "card_name_font_size", "Georgia", 75)
    font_cost = get_font_from_json("cost_font_family", "cost_font_size", "Chalkduster.ttf", 72)
    font_body = get_font_from_json("body_font_family", "body_font_size", ["Futura.ttc", "Avenir.ttc"], 22)
    font_abil_num = get_font_from_json("ability_num_font_family", "ability_num_font_size", ["Futura.ttc", "Avenir.ttc"], 24)
    font_abil_num_bold = get_font_from_json("ability_num_bold_font_family", "ability_num_bold_font_size", ["Futura-Bold.ttf", "Avenir-Heavy.ttf"], 25)
    font_flavor = get_font_from_json("flavor_text_font_family", "flavor_text_font_size", ["Georgia-Italic.ttf", "TimesNewRoman-Italic.ttf"], 20)

    # --- COLORS (Loaded from JSON) ---
    color_name = data.get("card_name_font_color", "#c8baa6")
    color_deploy_cost = data.get("deploy_cost_font_color", "white")
    color_body = data.get("body_font_color", "#F5F5DC")
    color_abil_cost = data.get("ability_cost_font_color", "black")
    letter_spacing_name = data.get("card_name_letter_spacing", 0)
    card_name_y_offset = data.get("card_name_y_offset", 0)
    icon_stack_x = data.get("icon_stack_x_offset", COST_POS_X)
    card_name_stroke_width = data.get("card_name_stroke_width", 0)
    card_name_stroke_color = data.get("card_name_stroke_color", "black")

    def resolve_art_path(art_name):
        """Return a usable path for the requested art and whether it is missing."""
        path = os.path.join(art_dir, art_name)
        if os.path.exists(path):
            return path, False
        if art_name.lower().endswith('.png'):
            fallback_path = os.path.join(art_dir, os.path.splitext(art_name)[0] + '.jpg')
            if os.path.exists(fallback_path):
                return fallback_path, False
        return path, True
    icon_stack_y_offset = data.get("icon_stack_y_offset", 0)
    
    def draw_wrapped_text(draw_context, text, start_pos, font, fill, indent=0, width=TEXT_WIDTH_CHARS):
        """Helper to draw wrapped text with an optional icon and return the new y-position."""
        x, y = start_pos
        lines = textwrap.wrap(text, width=width)
        line_height = font.getbbox("A")[3] + 5 # Get font height and add a small margin
        for i, line in enumerate(lines):
            line_indent = indent if i == 0 else 0
            draw_context.text((x + line_indent, y), line, font=font, fill=fill)
            y += line_height
        return y + 20

    generated_card_files = []
    
    # --- Prepare the list of cards to be rendered ---
    goons_to_render = []
    if create_grid and use_duplicates:
        print("   [+] '-dup' flag is active. Generating list based on 'duplicates' count.")
        for goon in data['goons']:
            num_copies = goon.get('duplicates', 1)
            if not isinstance(num_copies, int) or num_copies < 0:
                num_copies = 1
            for dup_index in range(num_copies):
                goons_to_render.append((goon, dup_index))
    else:
        # Default behavior: render one of each unique goon
        goons_to_render = [(goon, 0) for goon in data['goons']]

    prepared_art_cache = {}

    for i, (card_data, dup_index) in enumerate(goons_to_render):
        card = normalize_goon_data(card_data.copy(), faction, fix=fix) # Use a copy to avoid mutation issues
        name = card.get('name', 'Unnamed Goon') # Use .get for safety

        # Validate the schema for each card before processing
        schema_errors = validate_goon_schema(card)
        if schema_errors:
            print(f"   [!] ERROR: Card '{name}' failed schema validation. Skipping card.")
            for err in schema_errors:
                print(f"     - {err}")
            continue

        # Add a suffix for duplicate card filenames to avoid overwriting
        filename_suffix = ""
        if create_grid and use_duplicates and dup_index > 0:
            filename_suffix = f"_{dup_index + 1}"

        print(f"   [+] Processing: {name}{filename_suffix}")

        canvas = Image.new("RGBA", (750, 1050), (0, 0, 0, 255))
        
        # 1. Art
        portrait_list = card.get('portrait_art', [])
        art_records = prepared_art_cache.get(name)
        if art_records is None:
            art_records = []
            for portrait_name in portrait_list:
                resolved_path, missing = resolve_art_path(portrait_name)
                art_records.append({"name": portrait_name, "path": resolved_path, "missing": missing})

            # If -auto is set, generate any missing files in the portrait art array.
            if auto_generate_art and art_records:
                art_prompt = None
                for rec in art_records:
                    if not rec["missing"]:
                        continue
                    if art_prompt is None:
                        art_prompt = generate_art_prompt(card, ai_prompt_data, art_style_prompt)
                    if art_prompt:
                        success = generate_and_save_art(art_prompt, rec["path"], style_image=style_image)
                        if success and auto_extra_variations > 0:
                            root, ext = os.path.splitext(rec["path"])
                            for i in range(1, auto_extra_variations + 1):
                                alt_path = f"{root}_{i}{ext}"
                                generate_and_save_art(art_prompt, alt_path, style_image=style_image)
                        rec["missing"] = not success
                    else:
                        rec["missing"] = True

            prepared_art_cache[name] = art_records
        else:
            portrait_list = [rec["name"] for rec in art_records]

        selected_art_name = None
        if portrait_list:
            if create_grid and use_duplicates:
                selected_art_name = portrait_list[dup_index % len(portrait_list)]
            else:
                selected_art_name = portrait_list[0]

        art_file = None
        if selected_art_name:
            for rec in art_records:
                if rec["name"] == selected_art_name:
                    art_file = rec["path"]
                    break

        art_crop = None
        if art_file and os.path.exists(art_file):
            try:
                full_art = Image.open(art_file).convert("RGBA")
                
                # --- Resize and crop to fill the space while maintaining aspect ratio ("cover" style) ---
                target_w, target_h = 600, 585 # New art window size
                orig_w, orig_h = full_art.size
                
                target_aspect = target_w / target_h
                orig_aspect = orig_w / orig_h

                if orig_aspect > target_aspect:
                    # Original is wider than target: scale by height and crop width
                    new_h = target_h
                    new_w = int(new_h * orig_aspect)
                    resized_art = full_art.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    left = (new_w - target_w) / 2
                    art_crop = resized_art.crop((left, 0, left + target_w, target_h))
                else:
                    # Original is taller or same aspect: scale by width and crop height
                    new_w = target_w
                    new_h = int(new_w / orig_aspect)
                    resized_art = full_art.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    top = (new_h - target_h) / 2
                    art_crop = resized_art.crop((0, top, target_w, top + target_h))

            except Exception as e:
                print(f"     [!] Error processing art for {name}: {e}")
                art_crop = create_placeholder_art()
        else:
            missing_label = selected_art_name or art_file or "None"
            print(f"     [!] Art for {name} not found ('{missing_label}'). Using placeholder.")
            art_crop = create_placeholder_art()

        canvas.paste(art_crop, (130, 100))

        # 2. Frame
        canvas.alpha_composite(assets['frame'])

        draw = ImageDraw.Draw(canvas)
        
        # 3. Name
        name_bbox = draw.textbbox((0, 0), name, font=font_header)
        name_width = name_bbox[2] - name_bbox[0]
        start_x = NAME_END_X - name_width
        draw.text(
            (start_x, NAME_Y + card_name_y_offset), 
            name, 
            font=font_header, 
            fill=color_name, 
            spacing=letter_spacing_name,
            stroke_width=card_name_stroke_width,
            stroke_fill=card_name_stroke_color)

        # 4. Left Side Stack
        current_y = COST_START_Y + icon_stack_y_offset

        def is_cost_nonzero(val):
            if isinstance(val, str):
                return val.strip().upper() == "X"
            return isinstance(val, (int, float)) and val > 0

        def cost_label(val):
            if isinstance(val, str):
                cleaned = val.strip()
                return cleaned.upper() if cleaned else "0"
            try:
                return str(int(val))
            except Exception:
                return "0"
        
        def draw_main_icon(icon_key, value, y_pos, text_y_offset=0):
            icon = assets[icon_key]
            canvas.paste(icon, (icon_stack_x, y_pos), icon)
            # Use anchor="mm" for robust vertical and horizontal centering.
            draw.text((icon_stack_x + 42.5, y_pos + 42.5 + text_y_offset), cost_label(value), font=font_cost, fill=color_deploy_cost, anchor="mm")

        deploy_cost = card.get('deploy_cost', {})

        if is_cost_nonzero(deploy_cost.get('wind', 0)):
            draw_main_icon('cost_wind', deploy_cost['wind'], current_y, text_y_offset=-5)
            current_y += ICON_SPACING

        if is_cost_nonzero(deploy_cost.get('gear', 0)):
            draw_main_icon('cost_gear', deploy_cost['gear'], current_y, text_y_offset=-5)
            current_y += ICON_SPACING

        if is_cost_nonzero(deploy_cost.get('meat', 0)):
            draw_main_icon('cost_meat', deploy_cost['meat'], current_y, text_y_offset=-5)
            current_y += ICON_SPACING

        # --- 5. RANK ICON ---
        rank = card.get('rank')
        rank_map = {
            "SL": "rank_sl",
            "SG": "rank_sg",
            "T": "rank_t"
        }
        if rank in rank_map:
            icon_key = rank_map.get(rank)
            icon = assets[icon_key]
            canvas.paste(icon, (icon_stack_x, current_y), icon)
            current_y += ICON_SPACING


        traits = []
        if card.get('biological'): traits.append('trait_biological')
        if card.get('mechanical'): traits.append('trait_mechanical')
        if card.get('no_unwind'): traits.append('nounwind')
        if card.get('resist'): traits.append('resist')
        traits.append('faction') 

        for t in traits:
            icon = assets[t]
            canvas.paste(icon, (icon_stack_x, current_y), icon)
            current_y += ICON_SPACING

        # --- 6. ABILITIES ---
        text_y = TEXT_BOX_START_Y
        
        for ability in card['abilities']:
            icon_to_draw = None
            cost_text = None
            
            if ability.get('passive', False):
                icon_to_draw = assets['abil_passive'] 
                # Vertically center the passive icon with the first line of text
                font_height = font_body.getbbox("Test")[3] - font_body.getbbox("Test")[1]
                icon_height = icon_to_draw.height
                y_offset = (font_height - icon_height) // 2
                canvas.paste(icon_to_draw, (TEXT_BOX_START_X, text_y + y_offset + 5), icon_to_draw)
                indent = 40
            else:
                cost = ability.get('cost', {})
                wind = cost.get('wind', 0)
                meat = cost.get('meat', 0)
                gear = cost.get('gear', 0)
                wind_val = cost_label(wind)
                meat_val = cost_label(meat)
                gear_val = cost_label(gear)

                wind_cost_present = is_cost_nonzero(wind)
                meat_cost_present = is_cost_nonzero(meat)
                gear_cost_present = is_cost_nonzero(gear)

                # If all costs are 0, show '0' in a wind circle.
                is_zero_cost = not (wind_cost_present or meat_cost_present or gear_cost_present)

                # Build the cost parts to render left-to-right.
                cost_parts = []
                if wind_cost_present or is_zero_cost:
                    cost_parts.append(("wind", wind_val if wind_cost_present else "0"))
                if meat_cost_present:
                    cost_parts.append(("meat", meat_val))
                if gear_cost_present:
                    cost_parts.append(("gear", gear_val))

                x_cursor = TEXT_BOX_START_X
                indent = 0
                for idx, (kind, label) in enumerate(cost_parts):
                    if idx > 0:
                        draw.text((x_cursor, text_y), "+", font=font_body, fill=color_body)
                        x_cursor += 20

                    if kind == "wind":
                        icon_to_draw = create_circle_icon(30, "#F5F5DC")
                    elif kind == "meat":
                        icon_to_draw = create_circle_icon(30, "#8B0000") # Dark Red
                    else:
                        icon_to_draw = create_circle_icon(30, "#808080") # Grey

                    canvas.paste(icon_to_draw, (x_cursor, text_y + 2), icon_to_draw)
                    bbox = draw.textbbox((0, 0), label, font=font_abil_num_bold)
                    w_num, h_num = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    draw.text((x_cursor + 15 - w_num/2, text_y + 12 - h_num/2 - 2), label, font=font_abil_num_bold, fill=color_abil_cost)
                    x_cursor += 35

                indent = x_cursor - TEXT_BOX_START_X


            
            full_text = f"{ability['name'].upper()}: {ability['text']}"
            text_y = draw_wrapped_text(draw, full_text, (TEXT_BOX_START_X, text_y), font_body, color_body, indent=indent)

        # --- 7. DEPLOY REQUIREMENTS ---
        if 'deploy_requirements' in card:
            for req in card['deploy_requirements']:
                if req.get('type') == 'requires_card_in_play': 
                    icon_to_draw = assets['abil_star']
                    # Vertically center the star icon with the first line of text
                    font_height = font_body.getbbox("Test")[3] - font_body.getbbox("Test")[1]
                    icon_height = icon_to_draw.height
                    y_offset = (font_height - icon_height) // 2
                    canvas.paste(icon_to_draw, (TEXT_BOX_START_X, text_y + y_offset), icon_to_draw)
                    req_text = f"{req['card_name']} must be in play to deploy."
                    text_y = draw_wrapped_text(draw, req_text, (TEXT_BOX_START_X, text_y), font_body, color_body, indent=40)
        
        # --- 8. FLAVOR TEXT ---
        if card.get("flavor_text"):
            flavor_text = f'"{card["flavor_text"]}"'
            text_y = draw_wrapped_text(draw, flavor_text, (TEXT_BOX_START_X, text_y), font_flavor, color_body, indent=0, width=int(TEXT_WIDTH_CHARS * 1.4))

        filename = f"{output_dir}/{name.replace(' ', '_')}{filename_suffix}.png"
        canvas.save(filename)
        generated_card_files.append(filename)
    
    if fix:
        # Write the fixed data back to the JSON file
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\n--- FIXED {json_file} ---")


    print(f"\n--- SUCCESS ---")

    if create_grid:
        create_grid_image(generated_card_files, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Goon Squad Galaxy card images from JSON data.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-pcu', action='store_true', help="Process the PCU deck.")
    group.add_argument('-meat', action='store_true', help="Process the MEAT deck.")
    group.add_argument('-narc', action='store_true', help="Process the NARC deck.")
    group.add_argument('-omni', action='store_true', help="Process the OMNI deck.")
    group.add_argument('-necro', action='store_true', help="Process the NECRO deck.")
    parser.add_argument('-auto', nargs='?', const=0, type=int, help="Automatically generate missing portrait art. Optionally provide a number to generate that many additional variations (_1, _2, ...).")
    parser.add_argument('-deck', action='store_true', help="Render all cards in the selected deck.")
    parser.add_argument('-art', type=int, metavar='N', help="Generate N standalone art portraits using the selected faction's goon_traits/art_style; ignores deck rendering.")
    parser.add_argument('-grid', action='store_true', help="Generate a single grid image of all cards in the deck.")
    parser.add_argument('-dup', action='store_true', help="When using -grid, render multiple copies based on the 'duplicates' value.")
    parser.add_argument('-goon', nargs='?', const='__generate__', default=None, help="Generate a new goon definition. Optionally provide a name.")
    parser.add_argument('-fix', action='store_true', help="Automatically fix missing fields in the JSON data.")
    args = parser.parse_args()

    if args.pcu:
        deck_key = "pcu"
    elif args.narc:
        deck_key = "narc"
    elif args.meat:
        deck_key = "meat"
    elif args.omni:
        deck_key = "omni"
    elif args.necro:
        deck_key = "necro"
    else:
        raise ValueError("No deck selected; argparse should enforce one deck flag.")

    art_only_count = args.art
    if art_only_count is not None:
        generate_art_only_batch(
            faction=deck_key,
            art_dir=DECK_CONFIG[deck_key]["art"],
            count=art_only_count
        )
        sys.exit(0)

    deck_paths = DECK_CONFIG[deck_key]
    json_to_process = deck_paths["json"]
    art_directory = deck_paths["art"]
    output_directory = deck_paths["output"]
    faction_name = deck_key

    auto_requested = args.auto is not None
    auto_extra_variations = max(0, args.auto) if auto_requested else 0
    render_deck = args.deck or auto_requested

    if args.goon:
        goon_name = args.goon if args.goon != '__generate__' else None
        generate_new_goon(faction=faction_name, deck_json_path=json_to_process, goon_name=goon_name)
    elif render_deck:
        generate_cards(
            json_file=json_to_process, 
            art_dir=art_directory, 
            output_dir=output_directory, 
            faction=faction_name,
            auto_generate_art=auto_requested,
            auto_extra_variations=auto_extra_variations,
            create_grid=args.grid,
            use_duplicates=args.dup,
            fix=args.fix
        )
    else:
        print("[!] Deck rendering skipped. Pass '-deck' or '-auto' to render the selected deck.")
