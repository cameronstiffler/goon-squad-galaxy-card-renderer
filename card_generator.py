import json
import os
import sys
import textwrap
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
        font_path = None
        # First, check for a local file in the 'fonts' directory
        local_path = os.path.join("fonts", font_name)
        if os.path.exists(local_path):
            font_path = local_path
        else:
            # If not local, try to find the font on the system
            try:
                font_path = ImageFont.find_font(font_name)
            except Exception:
                font_path = font_name # Fallback to trying the name directly
        
        try:
            return ImageFont.truetype(font_path, size)
        except IOError:
            continue # Try the next candidate
    return default_font

def create_placeholder_art(size=(650, 600), text="ART MISSING"):
    """Creates a placeholder image for missing artwork."""
    img = Image.new('RGBA', size, (50, 50, 50, 255))
    draw = ImageDraw.Draw(img)
    font = get_font(["Georgia", "Arial"], 40)
    draw.text((size[0]/2, size[1]/2), text, font=font, anchor="mm", fill=(200, 200, 200))
    return img

def generate_art_prompt(goon_name, prompt_data):
    """Constructs a randomized, detailed prompt for DALL-E from a template."""
    if not prompt_data:
        print("     [!] WARNING: ai_art_prompt data not found in JSON. Skipping art generation.")
        return None

    template = prompt_data.get("template", "")
    options = prompt_data.get("options", {})

    prompt = template.format(
        goon_name=goon_name,
        armor=random.choice(options.get("armor", ["scrap"])),
        head=random.choice(options.get("head", ["a helmet"])),
        feet=random.choice(options.get("feet", ["boots"])),
        background=random.choice(options.get("background", ["a plain environment"]))
    )
    return prompt

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

def generate_cards(json_file, art_dir, output_dir, faction, auto_generate_art=False):
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
    ai_prompt_data = data.get("ai_art_prompt")
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

    for card in data['goons']:
        name = card['name']

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
                art_prompt = generate_art_prompt(name, ai_prompt_data)
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
        draw.text((start_x, NAME_Y), name, font=font_header, fill=color_name)

        # 4. Left Side Stack
        current_y = COST_START_Y
        
        def draw_main_icon(icon_key, value, y_pos, text_y_offset=0):
            icon = assets[icon_key]
            canvas.paste(icon, (COST_POS_X, y_pos), icon)
            # Use anchor="mm" for robust vertical and horizontal centering.
            draw.text((COST_POS_X + 42.5, y_pos + 42.5 + text_y_offset), str(value), font=font_cost, fill=color_deploy_cost, anchor="mm")

        if card['deploy_cost'].get('wind', 0) > 0:
            draw_main_icon('cost_wind', card['deploy_cost']['wind'], current_y, text_y_offset=-5)
            current_y += ICON_SPACING

        if card['deploy_cost'].get('gear', 0) > 0:
            draw_main_icon('cost_gear', card['deploy_cost']['gear'], current_y, text_y_offset=-5)
            current_y += ICON_SPACING

        if card['deploy_cost'].get('meat', 0) > 0:
            draw_main_icon('cost_meat', card['deploy_cost']['meat'], current_y, text_y_offset=-5)
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
            canvas.paste(icon, (COST_POS_X, current_y), icon)
            current_y += ICON_SPACING


        traits = []
        if card.get('biological'): traits.append('trait_biological')
        if card.get('mechanical'): traits.append('trait_mechanical')
        if card.get('no_unwind'): traits.append('nounwind')
        if card.get('resist'): traits.append('resist')
        traits.append('faction') 

        for t in traits:
            icon = assets[t]
            canvas.paste(icon, (COST_POS_X, current_y), icon)
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
                    canvas.paste(icon_to_draw, (x_cursor, text_y), icon_to_draw)
                    
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
                    canvas.paste(meat_icon, (x_cursor, text_y), meat_icon)
                    bbox = draw.textbbox((0, 0), str(meat), font=font_abil_num_bold)
                    w_num, h_num = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    draw.text((x_cursor + 15 - w_num/2, text_y + 12 - h_num/2 - 2), str(meat), font=font_abil_num_bold, fill=color_abil_cost)
                    x_cursor += 35
                    indent = x_cursor - TEXT_BOX_START_X
                elif gear > 0:
                    gear_icon = create_circle_icon(30, "#808080") # Grey
                    canvas.paste(gear_icon, (x_cursor, text_y), gear_icon)
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

    print(f"\n--- SUCCESS ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Goon Squad Galaxy card images from JSON data.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-pcu', action='store_true', help="Process the PCU deck.")
    group.add_argument('-meat', action='store_true', help="Process the MEAT deck.")
    group.add_argument('-narc', action='store_true', help="Process the NARC deck.")
    parser.add_argument('-auto', action='store_true', help="Automatically generate missing portrait art using DALL-E.")
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

    generate_cards(
        json_file=json_to_process, 
        art_dir=art_directory, 
        output_dir=output_directory, 
        faction=faction_name,
        auto_generate_art=args.auto
    )