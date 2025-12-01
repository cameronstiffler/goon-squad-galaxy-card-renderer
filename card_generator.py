import json
import os
import sys
import textwrap
from PIL import Image, ImageDraw, ImageFont

# === CONFIGURATION ===
SOURCE_ICONS = "all.png"          # Master Icon Sheet (Left side traits)
SOURCE_ABILITY_ICONS = "ability icons.png" # Ability Text Icons
FRAME_SOURCE = "frame.png"        # White-Center Frame
JSON_FILE = "pcu_deck_strict.json"
OUTPUT_DIR = "finished_cards"

IMAGE_MAP = {
    "Käsekobold": "goblin.jpg", 
    "Grim": "grim.jpg",
    "Helvis": "helvis.jpg",
    "Blood Howler": "howler.jpg",
    "Krax": "krax.jpg",
    "Kriegsgheist": "krieg.jpg",
    "Meatjacker": "meatjacker.jpg"
}

# --- COORDINATES ---
COST_POS_X = 40           
COST_START_Y = 55         
ICON_SPACING = 95         

NAME_END_X = 700  
NAME_Y = 25

TEXT_BOX_START = (75, 670)
TEXT_BOX_START_X, TEXT_BOX_START_Y = (75, 670)
TEXT_WIDTH_CHARS = 34     

def create_transparent_frame():
    print(f"   [+] Processing {FRAME_SOURCE}...")
    if not os.path.exists(FRAME_SOURCE):
        print(f"   [!] ERROR: Could not find {FRAME_SOURCE}.")
        sys.exit()

    img = Image.open(FRAME_SOURCE).convert("RGBA").resize((750, 1050))
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

def get_assets():
    print("--- STEP 1: PROCESSING ASSETS ---")
    if not os.path.exists(SOURCE_ICONS) or not os.path.exists(SOURCE_ABILITY_ICONS):
        print(f"[!] Critical Error: Missing icon files.")
        sys.exit()

    assets = {}
    assets['frame'] = create_transparent_frame()

    # --- 1. LEFT SIDE ICONS ---
    sheet = Image.open(SOURCE_ICONS).convert("RGBA")
    w, h = sheet.size
    row_h = h / 8 
    def get_main_icon(row_index, col=0):
        top = row_index * row_h
        bottom = top + row_h
        left = 0 if col == 0 else w / 2
        right = w / 2 if col == 0 else w
        icon = sheet.crop((left, top, right, bottom))
        bbox = icon.getbbox()
        if bbox: icon = icon.crop(bbox)
        return icon.resize((85, 85), Image.Resampling.LANCZOS)

    assets['cost_wind'] = get_main_icon(0, col=1)
    assets['cost_meat'] = get_main_icon(1, col=1)
    assets['cost_gear'] = get_main_icon(2, col=1)
    assets['resist'] = get_main_icon(3, col=0)
    assets['nounwind'] = get_main_icon(4, col=0)
    assets['trait_mechanical'] = get_main_icon(5, col=0) 
    assets['trait_biological'] = get_main_icon(6, col=0) 
    assets['faction'] = get_main_icon(7, col=0)

    # --- 2. ABILITY ICONS ---
    abil_sheet = Image.open(SOURCE_ABILITY_ICONS).convert("RGBA")
    aw, ah = abil_sheet.size
    a_row_h = ah / 3
    icon_size = int(a_row_h) 
    
    def get_abil_icon(row, col_index):
        top = row * a_row_h
        bottom = top + a_row_h
        left = col_index * icon_size
        right = left + icon_size
        icon = abil_sheet.crop((left, top, right, bottom))
        bbox = icon.getbbox()
        if bbox: icon = icon.crop(bbox)
        # UPDATED: Resize to 30x30 pixels
        return icon.resize((30, 30), Image.Resampling.LANCZOS)

    assets['abil_meat'] = get_abil_icon(2, 2)
    assets['abil_gear'] = get_abil_icon(2, 3)
    assets['abil_passive'] = get_abil_icon(2, 4)
    assets['abil_star'] = get_abil_icon(2, 5)

    print("   [+] Assets loaded.")
    return assets

def get_font(candidates, size, default_font=ImageFont.load_default()):
    """Tries to load a font from a list of candidates."""
    for font_name in candidates:
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

def generate_cards():
    print("\n--- STEP 2: GENERATING CARDS ---")
    with open(JSON_FILE, 'r') as f:
        data = json.load(f)

    assets = get_assets()
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    # --- FONTS ---
    try: font_header = ImageFont.truetype("Georgia", 75)
    except: font_header = ImageFont.load_default()
    font_header = get_font(["Georgia"], 75)
    font_cost = get_font(["Chalkduster.ttf", "Chalkduster.ttc"], 72)
    font_body = get_font(["Futura.ttc", "Futura.ttf", "Avenir.ttc", "GillSans.ttc", "Georgia"], 28)
    font_abil_num = get_font(["Futura.ttc", "Futura.ttf"], 18)

    def draw_wrapped_text(draw_context, text, start_pos, font, fill, icon=None):
        """Helper to draw wrapped text with an optional icon and return the new y-position."""
        x, y = start_pos
        indent = 0
        icon_width = 40

        if icon:
            draw_context.canvas.paste(icon, (x, y), icon)
            indent = icon_width

        lines = textwrap.wrap(text, width=TEXT_WIDTH_CHARS)
        for i, line in enumerate(lines):
            line_indent = indent if i == 0 else 0
            draw_context.text((x + line_indent, y), line, font=font, fill=fill)
            y += 35
        return y + 20

    for card in data['goons']:
        name = card['name']

        print(f"   [+] Processing: {name}")

        canvas = Image.new("RGBA", (750, 1050), (0, 0, 0, 255))
        
        # 1. Art
        art_file = card.get('portrait_art')
        art_crop = None
        if art_file and os.path.exists(art_file):
            try:
                full_art = Image.open(art_file).convert("RGBA")
                # Assuming a standard crop, this might need adjustment
                art_crop = full_art.crop((50, 100, 680, 650))
                art_crop = art_crop.resize((650, 600))
            except Exception as e:
                print(f"     [!] Error processing art for {name}: {e}")
                art_crop = create_placeholder_art()
        else:
            print(f"     [!] Art for {name} not found ('{art_file}'). Using placeholder.")
            art_crop = create_placeholder_art()

        canvas.paste(art_crop, (38, 110))

        # 2. Frame
        canvas.alpha_composite(assets['frame'])

        draw = ImageDraw.Draw(canvas)
        
        draw.canvas = canvas # Attach canvas to draw object for helper function
        # 3. Name
        name_bbox = draw.textbbox((0, 0), name, font=font_header)
        name_width = name_bbox[2] - name_bbox[0]
        start_x = NAME_END_X - name_width
        draw.text((start_x, NAME_Y), name, font=font_header, fill="#F5F5DC")

        # 4. Left Side Stack
        current_y = COST_START_Y
        
        def draw_main_icon(icon_key, value, y_pos, text_y_offset=0):
            icon = assets[icon_key]
            canvas.paste(icon, (COST_POS_X, y_pos), icon)
            val_str = str(value)
            bbox = draw.textbbox((0, 0), val_str, font=font_cost)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            draw.text((COST_POS_X + 42 - text_w/2, y_pos + 42 + text_y_offset - text_h/2 - 40), 
                      val_str, font=font_cost, fill="white")

        if card['deploy_cost'].get('wind', 0) > 0:
            draw_main_icon('cost_wind', card['deploy_cost']['wind'], current_y, text_y_offset=-10) 
            current_y += ICON_SPACING

        if card['deploy_cost'].get('gear', 0) > 0:
            draw_main_icon('cost_gear', card['deploy_cost']['gear'], current_y, text_y_offset=0)
            current_y += ICON_SPACING

        if card['deploy_cost'].get('meat', 0) > 0:
            draw_main_icon('cost_meat', card['deploy_cost']['meat'], current_y, text_y_offset=0)
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
            else:
                cost = ability.get('cost', {})
                wind = cost.get('wind', 0)
                meat = cost.get('meat', 0)
                gear = cost.get('gear', 0)
                wind_val = str(wind)
                
                if wind_val != "0":
                    # UPDATED: 30px Circle
                    icon_to_draw = create_circle_icon(30, "#F5F5DC")
                    cost_text = wind_val
                elif meat > 0:
                    icon_to_draw = assets['abil_meat']
                elif gear > 0:
                    icon_to_draw = assets['abil_gear']
            
            indent = 0
            if icon_to_draw:
                canvas.paste(icon_to_draw, (TEXT_BOX_START_X, text_y), icon_to_draw)
                
                if cost_text:
                    bbox = draw.textbbox((0, 0), cost_text, font=font_abil_num)
                    w_num = bbox[2] - bbox[0]
                    h_num = bbox[3] - bbox[1]
                    draw.text((TEXT_BOX_START_X + 15 - w_num/2, text_y + 15 - h_num/2 - 2), 
                              cost_text, font=font_abil_num, fill="black")
                
                indent = 40
            
            full_text = f"{ability['name'].upper()}: {ability['text']}"
            text_y = draw_wrapped_text(draw, full_text, (TEXT_BOX_START_X, text_y), font_body, "#F5F5DC", icon=icon_to_draw)

        # --- 7. DEPLOY REQUIREMENTS ---
        if 'deploy_requirements' in card:
            for req in card['deploy_requirements']:
                if req.get('type') == 'requires_card_in_play':
                    req_text = f"{req['card_name']} must be in play to deploy."
                    text_y = draw_wrapped_text(draw, req_text, (TEXT_BOX_START_X, text_y), font_body, "#F5F5DC", icon=assets['abil_star'])

        filename = f"{OUTPUT_DIR}/{name.replace(' ', '_')}.png"
        canvas.save(filename)

    print(f"\n--- SUCCESS ---")

if __name__ == "__main__":
    generate_cards()