import pandas as pd
import json
import random

def assign_ayurveda_properties(row):
    """
    Professional Ayurvedic Dosha Classification Engine
    Based on Charaka Samhita + Ashtanga Hridayam classical texts.
    
    Ayurvedic Principles Applied:
    - Rasa (Taste): Sweet/Sour/Salty pacify Vata; Pungent/Bitter/Astringent aggravate Vata
    - Vipaka (Post-digestive effect): Sweet = nourishing; Pungent = stimulating
    - Virya (Potency): Heating foods aggravate Pitta; Cooling pacify Pitta
    - Guna (Quality): Heavy/Oily aggravate Kapha; Light/Dry pacify Kapha
    """
    import re
    name = str(row['Dish Name']).lower()
    calories = float(row.get('KCAL', 0))
    fats = float(row.get('FAT (g)', 0))
    carbs = float(row.get('CARBS (g)', 0))
    protein = float(row.get('PROTEIN (g)', 0))

    # Default: Tridoshic balance (pacifies all) — override below
    v = "Pacifies"   # Vata
    pi = "Pacifies"  # Pitta
    k = "Pacifies"   # Kapha

    # ═══════════════════════════════════════════════════════════════
    # RULE SET 1: HEATING / PUNGENT / SPICY foods → AGGRAVATE PITTA
    # (Ushna Virya - hot potency foods increase heat, blood, inflammation)
    # ═══════════════════════════════════════════════════════════════
    pitta_heating_kws = [
        'chicken', 'mutton', 'lamb', 'pork', 'beef', 'meat', 'keema', 'kheema',
        'spicy', 'tikka', 'tandoori', 'pepper', 'chilli', 'chili', 'mirchi',
        'vinegar', 'tamarind', 'imli', 'rasam', 'pickle', 'achar',
        'garlic', 'onion', 'ginger', 'mustard',
        'tomato sauce', 'hot sauce', 'barbeque',
        'egg', 'fish', 'seafood', 'prawn', 'crab', 'salmon', 'tuna',
        'fermented', 'alcohol',
        'laal maas', 'rogan josh', 'vindaloo', 'chettinad',
        'achari', 'harissa',
        # Specific hot/sour Indian dishes
        'pav bhaji', 'misal', 'kolhapuri', 'schezwan', 'szechuan',
        'chole', 'chhole', 'rajma', 'dal makhani', 'makhani',
        'biryani', 'kachchi', 'kadhai',
        'pungent', 'sour', 'khatta', 'amchur', 'aamchur',
        'tomato', 'tamatar',  # tomato is sour/heating
        'lemon', 'nimbu', 'citrus',
        'ketchup', 'salsa', 'sriracha',
        'panch phoran', 'panch phoron',
        'dhansak', 'sorpotel', 'xacuti',  # Goan spicy dishes
        'szechwan', 'dragon', 'manchurian', 'chilli chicken',
    ]
    if any(kw in name for kw in pitta_heating_kws):
        pi = "Aggravates"

    # ═══════════════════════════════════════════════════════════════
    # RULE SET 2: HEAVY / SWEET / OILY / COLD foods → AGGRAVATE KAPHA
    # (Guru/Snigdha Guna - heavy and unctuous qualities increase Kapha)
    # ═══════════════════════════════════════════════════════════════
    kapha_heavy_kws = [
        'ice cream', 'milkshake', 'shake', 'cold coffee', 'cold drink',
        'butter', 'ghee', 'cream', 'cheese', 'paneer', 'khoya', 'khoa', 'mawa',
        'sweet', 'mithai', 'halwa', 'kheer', 'payasam', 'barfi', 'ladoo',
        'laddu', 'jalebi', 'gulab jamun', 'rasgulla', 'burfi', 'peda',
        'cake', 'pastry', 'cookie', 'biscuit', 'doughnut', 'bread',
        'naan', 'kulcha', 'pizza', 'burger', 'pasta', 'macaroni', 'noodle',
        'deep fried', 'fried', 'pakora', 'bhajia', 'vada', 'wada',
        'biryani', 'pulao',
        'condensed', 'coconut milk', 'full cream',
        'chocolate', 'caramel',
        'banana', 'avocado', 'dates', 'dry fruit',
    ]
    if any(kw in name for kw in kapha_heavy_kws):
        k = "Aggravates"
    # High-fat macro fallback (>20g fat = heavy for Kapha)
    if fats > 20:
        k = "Aggravates"

    # ═══════════════════════════════════════════════════════════════
    # RULE SET 3: DRY / COLD / RAW / BITTER / LIGHT foods → AGGRAVATE VATA
    # (Ruksha/Laghu/Sheeta Guna - dry, light, cold qualities disturb Vata)
    # ═══════════════════════════════════════════════════════════════
    vata_drying_kws = [
        'dry', 'raw', 'cold', 'ice', 'frozen',
        'salad', 'sprout', 'broccoli', 'cabbage', 'cauliflower', 'kale',
        'bitter gourd', 'karela', 'methi', 'fenugreek',
        'barley', 'corn', 'popcorn', 'millets',
        'legume', 'kidney bean', 'rajma', 'black bean',
        'chickpea', 'chana', 'lentil', 'moth bean',
        'astringent', 'unripe',
        'buttermilk', 'chaas',  # light, cold, drying
        'juice', 'nimbu', 'lemon',
    ]
    if any(kw in name for kw in vata_drying_kws):
        v = "Aggravates"

    # ═══════════════════════════════════════════════════════════════
    # RULE SET 4: AYURVEDIC POSITIVE OVERRIDES (Tridoshic foods)
    # Foods known to be universally balancing per Charaka Samhita
    # ═══════════════════════════════════════════════════════════════
    tridoshic_kws = [
        'rice', 'basmati', 'old rice', 'moong', 'mung',
        'pomegranate', 'coconut water', 'amla', 'gooseberry',
        'ghee',  # small amounts of ghee are tridoshic in Ayurveda
        'turmeric', 'saffron', 'cardamom',
        'milk', 'warm milk',
        'wheat', 'roti', 'chapati', 'khichdi', 'khichri',
        'sattu', 'triphala',
    ]
    if any(kw in name for kw in tridoshic_kws):
        # Reset only if not already strongly aggravating
        if v == "Aggravates" and k == "Pacifies" and pi == "Pacifies":
            v = "Pacifies"

    # ═══════════════════════════════════════════════════════════════
    # RULE SET 5: KAPHA-PACIFYING (Light, spicy, bitter foods)
    # Pungent/bitter/astringent rasa reduce Kapha
    # ═══════════════════════════════════════════════════════════════
    kapha_reducing_kws = [
        'ragi', 'finger millet', 'jowar', 'bajra', 'millet',
        'green leafy', 'spinach', 'palak', 'methi',
        'dal', 'sambhar', 'sambar', 'rasam',
        'idli', 'dosa', 'uttapam',  # fermented, light
        'soup', 'broth',
        'salad', 'raita',
        'moong dal', 'yellow dal', 'toor dal', 'arhar',
        'poha', 'upma', 'oats',
    ]
    if any(kw in name for kw in kapha_reducing_kws):
        if k == "Aggravates":
            k = "Pacifies"

    # ═══════════════════════════════════════════════════════════════
    # RULE SET 6: VATA-PACIFYING (Warm, oily, nourishing foods)
    # Sweet/Sour/Salty rasa and unctuous quality pacify Vata
    # ═══════════════════════════════════════════════════════════════
    vata_nourishing_kws = [
        'sesame', 'til', 'groundnut', 'peanut',
        'almond', 'cashew', 'walnut', 'pistachio',
        'sweet potato', 'yam', 'beet',
        'warm', 'hot', 'soup', 'stew',
        'ghee', 'oil', 'olive',
        'avial', 'sambar', 'dal',
        'dates', 'fig', 'raisin',
        'halwa', 'payasam', 'kheer',
    ]
    if any(kw in name for kw in vata_nourishing_kws):
        if v == "Aggravates":
            v = "Pacifies"

    # ═══════════════════════════════════════════════════════════════
    # RULE SET 7: PITTA-COOLING (Sweet, cooling, bitter foods)
    # Sweet/Bitter/Astringent rasa and Sheeta (cold) Virya pacify Pitta
    # ═══════════════════════════════════════════════════════════════
    pitta_cooling_kws = [
        'coconut', 'nariyal', 'cucumber', 'kheera',
        'mint', 'pudina', 'coriander', 'dhania',
        'fennel', 'saunf', 'cardamom', 'elaichi',
        'rose', 'gulab',
        'amla', 'pomegranate', 'anar',
        'green gram', 'moong', 'sattu',
        'curd', 'yogurt', 'raita',  # cooling but can aggravate Kapha
        'buttermilk', 'lassi',
        'barley water', 'watermelon',
    ]
    if any(kw in name for kw in pitta_cooling_kws):
        if pi == "Aggravates":
            pi = "Pacifies"

    # ═══════════════════════════════════════════════════════════════
    # RULE SET 8: MACRO-BASED FALLBACK
    # Very high-calorie dense foods aggravate Kapha by default
    # Very low-protein, dry foods can disturb Vata
    # ═══════════════════════════════════════════════════════════════
    if calories > 500 and k == "Pacifies":
        k = "Aggravates"
    if carbs > 70 and fats < 3 and v == "Pacifies":
        v = "Aggravates"  # high-carb, dry = Vata-aggravating

    return {
        "Vata": v,
        "Pitta": pi,
        "Kapha": k
    }


def assign_category(name, csv_cat):
    import re
    cat = str(csv_cat).lower()
    name_lower = str(name).lower()
    
    # ═══ PRIORITY 1: Name-based overrides (HIGHEST PRIORITY) ═══════
    # These override ANY CSV category because the name is the strongest signal.
    
    # 1a. Desserts by name (cakes, icings, puddings, Indian sweets, etc.)
    if 'icing' in name_lower or 'pudding' in name_lower or 'halwa' in name_lower or 'kheer' in name_lower or 'payasam' in name_lower or 'laddu' in name_lower or 'ladoo' in name_lower or 'barfi' in name_lower or 'burfi' in name_lower or 'jalebi' in name_lower or 'gulab jamun' in name_lower or 'custard' in name_lower:
        return 'Dessert'
    # Cakes and loafs (use word-boundary to avoid 'pancake')
    if re.search(r'\bcake\b', name_lower) or re.search(r'\bloaf\b', name_lower):
        return 'Dessert'
    
    # 1b. Condiments / Spice mixes by name (NOT standalone dishes)
    if re.search(r'\bmasala\b', name_lower) or 'podi' in name_lower or 'powder' in name_lower or 'marmalade' in name_lower or 'stock' in name_lower:
        return 'Side'
    
    # 1c. Beverages by name
    if 'lassi' in name_lower or 'smoothie' in name_lower or 'coffee' in name_lower or 'kaapi' in name_lower or 'tea' in name_lower or 'chai' in name_lower or 'lemonade' in name_lower or 'sherbet' in name_lower or 'squash' in name_lower or 'jigarthanda' in name_lower or 'thandai' in name_lower:
        return 'Beverage'
    
    # 1d. Breakfast Snacks by name (light single items for morning)
    if 'pancake' in name_lower or 'porridge' in name_lower or 'cereal' in name_lower:
        return 'Snack'
    
    # 1e. Staples by name (rice dishes, breads)
    if 'biryani' in name_lower or 'pulao' in name_lower or 'khichdi' in name_lower or 'khichri' in name_lower or 'rice' in name_lower or 'roti' in name_lower or 'rotte' in name_lower or 'naan' in name_lower or 'paratha' in name_lower or 'parantha' in name_lower or 'chapati' in name_lower:
        return 'Staple'
    
    # 1f. Heavy protein dishes = Main (even if CSV says Side/Snack)
    if 'paneer' in name_lower or 'chicken' in name_lower or 'fish' in name_lower or 'mutton' in name_lower or 'pork' in name_lower or 'beef' in name_lower or 'egg' in name_lower or 'dal' in name_lower or 'ham' in name_lower or 'meat' in name_lower or 'lamb' in name_lower or 'crab' in name_lower or 'prawn' in name_lower or 'shrimp' in name_lower or 'bacon' in name_lower or 'sausage' in name_lower:
        if 'egg nog' not in name_lower:
            return 'Main'
    
    # 1g. Sides by name
    if 'soup' in name_lower or 'salad' in name_lower or 'pickle' in name_lower or 'chutney' in name_lower or 'raita' in name_lower or 'sauce' in name_lower or 'jam' in name_lower:
        return 'Side'
    
    # ═══ PRIORITY 2: CSV Category fallback ═════════════════════════
    if 'beverage' in cat or 'drink' in cat or 'cooler' in cat or 'milkshake' in cat or 'lassi' in cat:
        return 'Beverage'
    if 'dessert' in cat or 'sweet' in cat or re.search(r'\bcake\b', cat) or 'cookie' in cat or 'ice cream' in cat:
        return 'Dessert'
    if 'bread' in cat or 'roti' in cat or 'rice' in cat or 'staple' in cat:
        return 'Staple'
    if 'snack' in cat or 'breakfast' in cat or 'bakery' in cat or 'sandwich' in cat or 'burger' in cat or 'tiffin' in cat:
        return 'Snack'
    if 'chutney' in cat or 'condiment' in cat or 'pickle' in cat or 'sauce' in cat or 'salad' in cat or 'soup' in cat or 'side' in cat or 'raita' in cat:
        return 'Side'
    if 'curry' in cat or 'main' in cat or 'dal' in cat or 'gravy' in cat:
        return 'Main'
    
    # ═══ PRIORITY 3: Default fallback ══════════════════════════════
    return 'Main'

def convert():
    df = pd.read_csv('Indian_Food_Nutrition_MASTER_Combined_VERIFIED.csv')
    
    # ═══ BLACKLIST: Raw ingredients & non-food items to REMOVE ═══
    BLACKLIST_KEYWORDS = [
        # Raw baking ingredients (not standalone dishes)
        'icing', 'glace icing', 'gum icing', 'royal icing', 'butter icing',
        'baking powder', 'baking soda', 'yeast', 'cornflour', 'arrowroot',
        'gelatine', 'gelatin', 'food colour', 'food color',
        'essence', 'vanilla extract', 'cocoa powder', 'cream of tartar',
        'suet', 'lard', 'dripping', 'shortening', 'marzipan', 'fondant',
        # Raw spice powders (you don't eat these alone)
        'garam masala', 'chat masala', 'tandoori masala',
        'rasam powder', 'sambar powder', 'curry powder',
        'chilli powder', 'turmeric powder', 'coriander powder',
        'cumin powder', 'mustard powder', 'pav bhaji masala',
        'karam podi', 'gun powder chutney', 'kobbari karam',
        # Stock/base ingredients / pure sauces not served alone
        'mixed stock', 'stock cube', 'bouillon', 'chocolate sauce', 'white sauce',
        # Non-Indian out-of-place items
        'aspic', 'tomato aspic', 'anchovy paste',
    ]
    
    db = []
    removed = []
    seen_names = set()
    
    for idx, row in df.iterrows():
        # Skip invalid rows
        if pd.isna(row['KCAL']) or row['KCAL'] == 0:
            continue
        
        name = str(row['Dish Name'])
        name_lower = name.lower()
        
        # Check blacklist
        is_blacklisted = False
        for kw in BLACKLIST_KEYWORDS:
            if kw in name_lower:
                is_blacklisted = True
                removed.append(f"[Blacklist] {name} (matched: {kw})")
                break
        if is_blacklisted:
            continue
        
        # Assign category
        category = assign_category(name, row['Category'])
        
        # Remove ultra-low calorie items in meal categories (likely spices, not food)
        cal = float(row['KCAL'])
        if cal < 15 and category in ['Main', 'Snack', 'Staple']:
            removed.append(f"[Low-cal] {name} ({cal}kcal as {category})")
            continue
        
        # Remove duplicates (keep first occurrence)
        if name in seen_names:
            removed.append(f"[Duplicate] {name}")
            continue
        seen_names.add(name)
            
        db.append({
            "id": len(db) + 1,
            "name": name,
            "state_region": str(row['State/Region']) if not pd.isna(row['State/Region']) else "Unknown",
            "category": category,
            "diet": str(row['Diet']) if not pd.isna(row['Diet']) else "Mixed / Both",
            "calories": cal,
            "protein": float(row['PROTEIN (g)']),
            "carbs": float(row['CARBS (g)']),
            "fats": float(row['FAT (g)']),
            "ayurveda": {
                "dosha_effect": assign_ayurveda_properties(row),
                "notes": str(row['Ingredients with Ayurvedic Notes']) if not pd.isna(row['Ingredients with Ayurvedic Notes']) else "No specific Ayurvedic notes available."
            }
        })
        
    with open('full_nutritional_db.json', 'w') as f:
        json.dump(db, f, indent=2)
    
    # Print stats
    cats = {}
    for item in db:
        c = item['category']
        cats[c] = cats.get(c, 0) + 1
    
    print(f"=== AyurNutri Food DB Cleanup Report ===")
    print(f"  CSV rows processed: {len(df)}")
    print(f"  Items REMOVED: {len(removed)}")
    for r in removed:
        print(f"    - {r}")
    print(f"  Clean items saved: {len(db)}")
    print(f"  Category breakdown:")
    for c, n in sorted(cats.items()):
        print(f"    {c}: {n}")
    print(f"  Saved to full_nutritional_db.json")

if __name__ == "__main__":
    convert()

