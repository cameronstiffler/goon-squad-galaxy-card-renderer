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

# === CONFIGURATION ===
SOURCE_ICONS = "all.png"          # Master Icon Sheet (Left side traits)
SOURCE_ABILITY_ICONS = "ability icons.png" # Ability Text Icons
OUTPUT_DIR = "finished_cards"

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

def load_art_style_prompt():
    """Loads and formats the art style description from art_style.json."""
    try:
        with open('art_style.json', 'r') as f:
            style_data = json.load(f)
        return style_data.get("art_style_description", "A character illustration.")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"   [!] WARNING: Could not load art_style.json. Using default prompt. Error: {e}")
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

def generate_art_prompt(goon_name, prompt_data, art_style_prompt):
    """Constructs a randomized, detailed prompt for DALL-E using a base style and character template."""
    if not prompt_data:
        print("     [!] WARNING: ai_art_prompt data not found in JSON. Skipping art generation.")
        return None

    template = prompt_data.get("template", "")
    options = prompt_data.get("options", {})
    quantities = prompt_data.get("option_trait_quantities", {})

    if isinstance(template, list):
        template = "\n".join(template)
    elif not isinstance(template, str):
        template = str(template)

    format_args = {'goon_name': goon_name}

    placeholder_fields = {match for match in re.findall(r"{([^}]+)}", template)}
    placeholder_fields.discard('goon_name')

    for field in placeholder_fields:
        if field in options and options[field]:
            # Use the quantity specified in option_trait_quantities, default to 1
            quantity = quantities.get(field, 1)
            # Ensure we don't request more options than are available
            quantity = min(quantity, len(options[field]))
            selected_options = random.sample(options[field], quantity)
            format_args[field] = ", ".join(selected_options)
        else:
            format_args[field] = f"default_{field}"

    character_description = template.format(**format_args)

    priority_note = "Use the above art style exactly. Do not introduce any new style cues; only apply the subject details below."
    return f"{art_style_prompt}\n\n{priority_note}\n\nSubject: {character_description}"

def generate_and_save_art(prompt, save_path):
    """Generates art using DALL-E and saves it to the specified path."""
    print(f"     [+] Generating AI art for: {os.path.basename(save_path)}...")
    print(f"     [+] Using Prompt: {prompt}")
    try:
        client = openai.OpenAI() # This line reads the key from the environment variable
        response = client.images.generate(
            model="dall-e-3",
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

def create_grid_image(card_files, output_folder):
    """Arranges all generated cards into a single grid image."""
    if not card_files:
        print("\n--- No cards to generate grid. ---")
        return

    print("\n--- STEP 3: GENERATING GRID IMAGE ---")

    # Grid layout settings
    COLS = 5
    TARGET_GRID_WIDTH = 4500
    
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
        goon.setdefault('portrait_art', "")
        
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

    abilities = goon.get("abilities", [])
    if not isinstance(abilities, list):
        errors.append("abilities should be a list")
        return errors

    for ability in abilities:
        for key in required_ability_keys:
            if key not in ability:
                errors.append(f"ability '{ability.get('name', '?')}' missing field '{key}'")

    return errors

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
5.  Give it a unique `portrait_art` filename ending in `.jpg`.
"""

    # 3. Call the AI to generate the goon JSON
    print("   [+] Prompting AI to generate new goon...")
    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
        )
        goon_json_string = response.choices[0].message.content

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

def generate_cards(json_file, art_dir, output_dir, faction, auto_generate_art=False, create_grid=False, fix=False):
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
    art_style_prompt = load_art_style_prompt() # Load the master art style once
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
    for card in data['goons']:
        card = normalize_goon_data(card, faction, fix=fix)
        name = card.get('name', 'Unnamed Goon') # Use .get for safety

        # Validate the schema for each card before processing
        schema_errors = validate_goon_schema(card)
        if schema_errors:
            print(f"   [!] ERROR: Card '{name}' failed schema validation. Skipping card.")
            for err in schema_errors:
                print(f"     - {err}")
            continue

        print(f"   [+] Processing: {name}")

        canvas = Image.new("RGBA", (750, 1050), (0, 0, 0, 255))
        
        # 1. Art
        art_filename = card.get('portrait_art')
        art_file = None
        if art_filename:
            art_file = os.path.join(art_dir, art_filename)

            # If the primary art file doesn't exist, try swapping the extension
            if not os.path.exists(art_file):
                if art_filename.lower().endswith('.png'):
                    fallback_filename = os.path.splitext(art_filename)[0] + '.jpg'
                    fallback_path = os.path.join(art_dir, fallback_filename)
                    if os.path.exists(fallback_path):
                        art_file = fallback_path
            
            # If art still doesn't exist and auto-gen is on, create it
            if not os.path.exists(art_file) and auto_generate_art:
                art_prompt = generate_art_prompt(name, ai_prompt_data, art_style_prompt)
                if art_prompt:
                    if not generate_and_save_art(art_prompt, art_file):
                        art_file = None # Fallback to placeholder if generation fails
                else: art_file = None

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
            print(f"     [!] Art for {name} not found ('{art_file}'). Using placeholder.")
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
        
        def draw_main_icon(icon_key, value, y_pos, text_y_offset=0):
            icon = assets[icon_key]
            canvas.paste(icon, (icon_stack_x, y_pos), icon)
            # Use anchor="mm" for robust vertical and horizontal centering.
            draw.text((icon_stack_x + 42.5, y_pos + 42.5 + text_y_offset), str(value), font=font_cost, fill=color_deploy_cost, anchor="mm")

        deploy_cost = card.get('deploy_cost', {})

        if deploy_cost.get('wind', 0) > 0:
            draw_main_icon('cost_wind', deploy_cost['wind'], current_y, text_y_offset=-5)
            current_y += ICON_SPACING

        if deploy_cost.get('gear', 0) > 0:
            draw_main_icon('cost_gear', deploy_cost['gear'], current_y, text_y_offset=-5)
            current_y += ICON_SPACING

        if deploy_cost.get('meat', 0) > 0:
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
                wind_val = str(wind)
                
                # If all costs are 0, show '0' in a wind circle.
                is_zero_cost = wind == 0 and meat == 0 and gear == 0 and wind_val != "X"

                # --- NEW: Handle Combined Costs ---
                x_cursor = TEXT_BOX_START_X
                indent = 0
                
                # 1. Draw Wind Cost (if it exists)
                if (isinstance(wind, int) and wind > 0) or wind_val == "X" or is_zero_cost:
                    icon_to_draw = create_circle_icon(30, "#F5F5DC")
                    canvas.paste(icon_to_draw, (x_cursor, text_y + 2), icon_to_draw)
                    
                    bbox = draw.textbbox((0, 0), wind_val, font=font_abil_num_bold)
                    w_num, h_num = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    draw.text((x_cursor + 15 - w_num/2, text_y + 12 - h_num/2 - 2), wind_val, font=font_abil_num_bold, fill=color_abil_cost)
                    x_cursor += 35 # Move cursor past the icon
                    indent = x_cursor - TEXT_BOX_START_X

                # 2. Draw "+" if there's a second cost
                if (isinstance(wind, int) and wind > 0) and (meat > 0 or gear > 0):
                    draw.text((x_cursor, text_y), "+", font=font_body, fill=color_body)
                    x_cursor += 20 # Move cursor past the "+"
                    indent = x_cursor - TEXT_BOX_START_X

                # 3. Draw Meat or Gear cost
                if meat > 0:
                    meat_icon = create_circle_icon(30, "#8B0000") # Dark Red
                    canvas.paste(meat_icon, (x_cursor, text_y + 2), meat_icon)
                    bbox = draw.textbbox((0, 0), str(meat), font=font_abil_num_bold)
                    w_num, h_num = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    draw.text((x_cursor + 15 - w_num/2, text_y + 12 - h_num/2 - 2), str(meat), font=font_abil_num_bold, fill=color_abil_cost)
                    x_cursor += 35
                    indent = x_cursor - TEXT_BOX_START_X
                elif gear > 0:
                    gear_icon = create_circle_icon(30, "#808080") # Grey
                    canvas.paste(gear_icon, (x_cursor, text_y + 2), gear_icon)
                    bbox = draw.textbbox((0, 0), str(gear), font=font_abil_num_bold)
                    w_num, h_num = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    draw.text((x_cursor + 15 - w_num/2, text_y + 12 - h_num/2 - 2), str(gear), font=font_abil_num_bold, fill=color_abil_cost)
                    x_cursor += 35
                    indent = x_cursor - TEXT_BOX_START_X
                
                # Fallback for single meat/gear cost (if wind is 0)
                elif not indent and (meat > 0 or gear > 0):
                    # This case is now handled by the logic above, but we can keep it as a safeguard
                    # if the combined cost logic were to be removed. For now, it's redundant.
                    pass


            
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

        filename = f"{output_dir}/{name.replace(' ', '_')}.png"
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
    parser.add_argument('-auto', action='store_true', help="Automatically generate missing portrait art using DALL-E.")
    parser.add_argument('-grid', action='store_true', help="Generate a single grid image of all cards in the deck.")
    parser.add_argument('-goon', nargs='?', const='__generate__', default=None, help="Generate a new goon definition. Optionally provide a name.")
    parser.add_argument('-fix', action='store_true', help="Automatically fix missing fields in the JSON data.")
    args = parser.parse_args()

    if args.pcu:
        json_to_process = "pcu_deck_strict.json"
        art_directory = "art/pcu"
        faction_name = "pcu"
        output_directory = os.path.join(OUTPUT_DIR, "pcu")
    elif args.narc:
        json_to_process = "narc_deck_strict.json"
        art_directory = "art/narc"
        faction_name = "narc"
        output_directory = os.path.join(OUTPUT_DIR, "narc")
    elif args.meat:
        json_to_process = "meat_deck_strict.json"
        art_directory = "art/meat"
        faction_name = "meat"
        output_directory = os.path.join(OUTPUT_DIR, "meat")

    if args.goon:
        goon_name = args.goon if args.goon != '__generate__' else None
        generate_new_goon(faction=faction_name, deck_json_path=json_to_process, goon_name=goon_name)
    else:
        generate_cards(
            json_file=json_to_process, 
            art_dir=art_directory, 
            output_dir=output_directory, 
            faction=faction_name,
            auto_generate_art=args.auto,
            create_grid=args.grid,
            fix=args.fix
        )
