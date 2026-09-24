import os
import json
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
from neural_nlp import generate_neural_recipe, generate_neural_chat

try:
    from groq import Groq
except ImportError:
    Groq = None

from fastapi import Depends, HTTPException, status, Security
from fastapi.security import APIKeyHeader

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_api_key(api_key_header: str = Security(api_key_header)):
    # In production (Render), set AYURNUTRI_API_KEY in environment variables
    expected_api_key = os.getenv("AYURNUTRI_API_KEY", "dev-secret-key")
    if api_key_header == expected_api_key:
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key",
    )

app = FastAPI(
    title="AyurNutri Hybrid AI API", 
    version="1.0.0",
    dependencies=[Depends(get_api_key)]
)

# Enable CORS so the frontend can talk to the backend from anywhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Initialize API Keys
load_dotenv()



# Initialize Groq
groq_client = None
if os.getenv("GROQ_API_KEY"):
    if Groq:
        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        print("Groq API enabled!")
    else:
        print("Warning: GROQ_API_KEY found but groq library is not installed.")

# 2. Load ML Models
try:
    dosha_model = joblib.load("dosha_classifier.pkl")
    model_columns = joblib.load("model_columns.pkl")
    print("Successfully loaded ML models!")
except Exception as e:
    dosha_model = None
    model_columns = None
    print(f"Warning: ML models not found. ({e})")

# 3. Load Nutritional Database (now using the 1014-item DB)
try:
    # Reload trigger v3.6 (ayurveda-engine-v2.1-pitta-fix)
    with open("full_nutritional_db.json", "r") as f:
        nutritional_db = json.load(f)
except Exception as e:
    nutritional_db = []
    print(f"Warning: full_nutritional_db.json not found. ({e})")

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Ensure static directory exists
if not os.path.exists("static"):
    os.makedirs("static")
    
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    # Return the frontend dashboard
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {"message": "Welcome to the AyurNutri Hybrid AI API"}

# ---------------------------------------------------------
# Pipeline A: Dosha ML Classification
# ---------------------------------------------------------
class DoshaAssessmentRequest(BaseModel):
    body_frame: str
    walk_and_talk: str
    weather_reaction: str
    sweating: str
    appetite: str
    skin: str
    hair: str
    lips_and_teeth: str
    eyes: str
    general_signs: str
    memory: str
    mind: str
    mind_on_actions: str
    sleep_quality: str
    emotional_nature: str

@app.post("/api/assess-dosha/")
def assess_dosha(data: DoshaAssessmentRequest):
    if dosha_model is None or model_columns is None:
        return {"status": "error", "message": "ML Model is not trained yet."}

    input_data = pd.DataFrame([{
        "body_frame": data.body_frame,
        "walk_and_talk": data.walk_and_talk,
        "weather_reaction": data.weather_reaction,
        "sweating": data.sweating,
        "appetite": data.appetite,
        "skin": data.skin,
        "hair": data.hair,
        "lips_and_teeth": data.lips_and_teeth,
        "eyes": data.eyes,
        "general_signs": data.general_signs,
        "memory": data.memory,
        "mind": data.mind,
        "mind_on_actions": data.mind_on_actions,
        "sleep_quality": data.sleep_quality,
        "emotional_nature": data.emotional_nature
    }])

    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    probabilities = dosha_model.predict_proba(input_encoded)[0]
    classes = dosha_model.classes_
    
    prob_dict = {classes[i]: round(float(probabilities[i]), 2) for i in range(len(classes))}
    
    # Sort doshas by probability descending
    sorted_doshas = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    d1_name, d1_prob = sorted_doshas[0]
    d2_name, d2_prob = sorted_doshas[1]
    d3_name, d3_prob = sorted_doshas[2]
    
    # Determine dual doshas based on Ayurvedic principles (if primary and secondary are close)
    # If the difference between the top 2 is less than or equal to 0.15 (15%), it's a dual dosha
    if (d1_prob - d2_prob) <= 0.15:
        # Sort the dual dosha alphabetically for consistency (e.g., Pitta-Kapha)
        # or use specific Ayurvedic standard combinations
        combo = sorted([d1_name, d2_name])
        dominant = f"{combo[0]}-{combo[1]}"
        if combo == ["Kapha", "Pitta"]:
            dominant = "Pitta-Kapha"
        elif combo == ["Pitta", "Vata"]:
            dominant = "Vata-Pitta"
        elif combo == ["Kapha", "Vata"]:
            dominant = "Vata-Kapha"
            
        # Check for Tridosha (all 3 are very close)
        if (d1_prob - d3_prob) <= 0.10:
            dominant = "Tridosha"
    else:
        dominant = d1_name

    return {
        "status": "success",
        "predicted_dosha_probabilities": prob_dict,
        "dominant_dosha": dominant
    }

# ---------------------------------------------------------
# Pipeline B: Meal Optimization (Constraint Satisfaction)
# ---------------------------------------------------------
class MealPlanRequest(BaseModel):
    dosha: str
    target_calories: int = 2000
    target_protein: int = 70
    target_carbs: int = 200
    target_fat: int = 60
    cuisine: str = "Any"
    
    # New profile fields
    dietary_preference: str = "Mixed / Both"
    goal: str = "Balance Doshas"
    weight_kg: float = None
    height_cm: float = None
    age: int = None
    gender: str = None

@app.post("/api/generate-meal/")
def generate_meal_plan(data: MealPlanRequest):
    """
    ╔══════════════════════════════════════════════════════════════════╗
    ║  AyurNutri v4.0 — Multi-Objective Greedy Constraint Solver       ║
    ║  Algorithm: Greedy Best-Fit with Residual Correction             ║
    ║  + Diversity Enforcement + Dosha/Diet-Aware Filtering            ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    if not nutritional_db:
        return {"status": "error", "message": "Nutritional Database is empty."}
    
    import random
    import time
    
    start_time = time.time()
    print("═══ AyurNutri Constraint Solver v4.0 Started ═══")
    
    # ── PHASE 0: Dynamic Target Calculation (BMR & Goal) ───────────
    target_cal = data.target_calories
    target_pro = data.target_protein
    target_carb = data.target_carbs
    target_fat = data.target_fat
    
    if data.weight_kg and data.height_cm and data.age and data.gender:
        # Mifflin-St Jeor Equation (Base BMR)
        if data.gender.lower() == "male":
            bmr = 10 * data.weight_kg + 6.25 * data.height_cm - 5 * data.age + 5
        else:
            bmr = 10 * data.weight_kg + 6.25 * data.height_cm - 5 * data.age - 161
            
        tdee = bmr * 1.375  # Assuming light activity by default
        
        # Adjust based on goal
        if data.goal == "Lose Weight":
            target_cal = int(tdee - 500)
            target_pro = int((target_cal * 0.25) / 4) # 25% protein
        elif data.goal == "Gain Weight" or data.goal == "Build Muscle":
            target_cal = int(tdee + 500)
            target_pro = int((target_cal * 0.30) / 4) # 30% protein
        else: # Balance Doshas / Boost Energy
            target_cal = int(tdee)
            target_pro = int((target_cal * 0.20) / 4) # 20% protein
            
        target_carb = int((target_cal * 0.50) / 4) # 50% carbs
        target_fat = int((target_cal * 0.25) / 9)  # 25% fat
        
    print(f"  Targets: {target_cal} kcal, Pro:{target_pro}g, Carb:{target_carb}g, Fat:{target_fat}g")
    
    # ── PHASE 1: Filtering (Dosha + Diet + Cuisine) ──────────
    pools = {
        "Main": [], "Side": [], "Beverage": [],
        "Staple": [], "Snack": [], "Dessert": []
    }
    
    for food in nutritional_db:
        # 1. Diet filter
        food_diet = food.get("diet", "Mixed / Both")
        pref = data.dietary_preference
        if pref == "Vegetarian" and food_diet == "Non-Veg":
            continue
        if pref == "Vegan":
            if food_diet != "Veg":
                continue
            # Basic vegan filter: exclude items with milk, ghee, paneer, curd, etc.
            name_low = food["name"].lower()
            if any(dairy in name_low for dairy in ['milk', 'ghee', 'paneer', 'curd', 'yogurt', 'cheese', 'butter', 'cream', 'khoya']):
                continue
        
        # 2. Cuisine filter
        if data.cuisine != "Any":
            region = food.get("state_region", "Unknown")
            if not (data.cuisine in region or "Pan-India" in region or "Continental/Global" in region):
                continue
        
        # 3. Dosha filter
        # Handle Dual Doshas (e.g., Pitta-Kapha)
        doshas_to_check = data.dosha.split("-")
        aggravates = False
        for d in doshas_to_check:
            effect = food.get("ayurveda", {}).get("dosha_effect", {}).get(d, "Unknown")
            if effect == "Aggravates":
                aggravates = True
                break
        if aggravates:
            continue
            
        cat = food.get("category", "Main")
        if cat in pools:
            pools[cat].append(food)
    
    # Build Pan-India fallback pools (activated when regional pool exhausts mid-week)
    pan_india_pools = {k: [] for k in pools}
    for food in nutritional_db:
        region = food.get("state_region", "")
        if "Pan-India" in region or "Continental/Global" in region:
            # Dual dosha fallback check
            doshas_to_check = data.dosha.split("-")
            aggravates = False
            for d in doshas_to_check:
                effect = food.get("ayurveda", {}).get("dosha_effect", {}).get(d, "Unknown")
                if effect == "Aggravates":
                    aggravates = True
                    break
            if not aggravates:
                cat = food.get("category", "Main")
                if cat in pan_india_pools:
                    pan_india_pools[cat].append(food)
    
    # If any regional pool starts empty, fill from Pan-India first, then all cuisines
    for cat_name, cat_list in pools.items():
        if not cat_list:
            fallback = pan_india_pools.get(cat_name, [])
            if fallback:
                cat_list.extend(fallback)
            else:
                for food in nutritional_db:
                    if food.get("category") == cat_name:
                        effect = food.get("ayurveda", {}).get("dosha_effect", {}).get(data.dosha, "Unknown")
                        if effect != "Aggravates":
                            cat_list.append(food)
    
    print(f"  Pools: Main={len(pools['Main'])}, Staple={len(pools['Staple'])}, "
          f"Side={len(pools['Side'])}, Snack={len(pools['Snack'])}, "
          f"Bev={len(pools['Beverage'])}, Dessert={len(pools['Dessert'])}")
    
    # ── PHASE 2: Greedy Best-Fit Selector ──────────────────────────
    def compute_loss(food, target_cal, target_pro, target_carb, target_fat):
        """
        4-Variable Weighted Loss Function.
        Lower = better fit to the target macros.
        Weights: Calories (1.5x priority) > Protein = Carbs = Fat (1.0x each)
        """
        cal = food.get("calories", 0)
        pro = food.get("protein", 0)
        carb = food.get("carbs", 0)
        fat = food.get("fats", 0)
        loss = (
            (abs(cal - target_cal) / max(target_cal, 1)) * 1.5 +
            (abs(pro - target_pro) / max(target_pro, 1)) * 1.0 +
            (abs(carb - target_carb) / max(target_carb, 1)) * 1.0 +
            (abs(fat - target_fat) / max(target_fat, 1)) * 1.0
        )
        return loss
    
    def pick_best(pool, target_cal, target_pro, target_carb, target_fat, used_ids, max_cal_ratio=3.0, day_seed=0, exclude_heavy=False):
        """
        Weighted-Random Best-Fit Selector with Daily Variety.
        
        Algorithm:
        1. Compute loss for all eligible foods
        2. Build a 'competitive set': all foods within 60% of the best loss
        3. Randomly select from the competitive set using day_seed
        """
        import random
        cal_ceiling = target_cal * max_cal_ratio
        
        # Heavy keywords that shouldn't be eaten at dinner
        heavy_kws = ["pulao", "biryani", "paneer", "meat", "chicken", "mutton", "beef", "pork"]
        
        # Step 1: Score all eligible foods
        scored = []
        for food in pool:
            fid = food.get("id", food.get("name"))
            if fid in used_ids:
                continue
            if exclude_heavy and any(kw in food.get("name", "").lower() for kw in heavy_kws):
                continue
            if food.get("calories", 0) > cal_ceiling and target_cal > 0:
                continue
            loss = compute_loss(food, target_cal, target_pro, target_carb, target_fat)
            scored.append((loss, food))
        
        if not scored:
            return None
        
        # Step 2: Build competitive set (within 60% of best loss)
        scored.sort(key=lambda x: x[0])
        best_loss = scored[0][0]
        # Allow any food whose loss is within 60% above the best 
        # (e.g., if best=0.45, include all with loss <= 0.72)
        threshold = best_loss * 1.6 + 0.1  # +0.1 ensures at least 1 item when best_loss=0
        competitive = [food for loss, food in scored if loss <= threshold]
        
        # Step 3: Weighted random pick from competitive set using day seed only
        random.seed(day_seed * 37)
        return random.choice(competitive)
    
    # ── PHASE 3: 7-Day Plan Generation with Residual Correction ────
    #
    # Meal Structure (like a real human eats):
    #   Breakfast      → 1 Snack item (Dosa, Idli, Poha, Upma...)     25%
    #   Mid-Morning    → 1 Beverage (Coffee, Lassi, Juice...)           5%
    #   Lunch          → Staple + Main + Side (Rice + Curry + Chutney) 40%
    #   Evening Snack  → 1 Snack item (Dhokla, Murukku, Pakora...)    10%
    #   Dinner         → Staple + Main (Roti + Dal)                    20%
    #
    # Residual Correction: If Breakfast overshoots target by +X kcal,
    # the remaining meals' targets are reduced proportionally so the
    # daily total still hits the exact target.
    
    MEAL_STRUCTURE = [
        ("Breakfast",        [("Snack",)],                          0.25),
        ("Mid_Morning_Snack",[("Beverage",)],                       0.05),
        ("Lunch",            [("Staple",), ("Main",), ("Side",)],   0.40),
        ("Evening_Snack",    [("Snack",)],                          0.10),
        ("Dinner",           [("Staple",), ("Main",)],              0.20),
    ]
    
    # ── Regional Food Pins ──────────────────────────────────────────
    # Guarantee that iconic regional dishes appear in the plan.
    # Format: {cuisine: [(day, meal, food_name_substring), ...]}
    REGIONAL_PINS = {
        "Karnataka": [
            (3, "Dinner",   "Mudde"),          # Ragi Mudde on Day 3 Dinner
            (6, "Dinner",   "Jolada Rotti"),   # Jolada Rotti on Day 6 Dinner
        ],
        "Kerala": [
            (2, "Dinner",   "Puttu"),          # Puttu on Day 2 Dinner
            (5, "Breakfast","Appam"),          # Appam on Day 5 Breakfast
        ],
        "Tamil Nadu": [
            (2, "Breakfast","Idli"),           # Idli on Day 2 Breakfast
            (5, "Lunch",   "Sambar"),          # Sambar on Day 5 Lunch
        ],
        "Rajasthan": [
            (3, "Lunch",   "Dal Baati"),       # Dal Baati on Day 3
            (6, "Dinner",  "Laal Maas"),       # Laal Maas on Day 6
        ],
    }
    # Pre-pin: find and lock regional foods into their slots
    pinned_foods = {}  # {(day, meal): food_object}
    if data.cuisine in REGIONAL_PINS:
        for (pin_day, pin_meal, pin_substr) in REGIONAL_PINS[data.cuisine]:
            pin_match = None
            for food in nutritional_db:
                if pin_substr.lower() in food["name"].lower():
                    region = food.get("state_region", "")
                    effect = food.get("ayurveda", {}).get("dosha_effect", {}).get(data.dosha, "Unknown")
                    if effect != "Aggravates":
                        pin_match = food
                        break
            if pin_match:
                pinned_foods[(pin_day, pin_meal)] = pin_match
    
    weekly_plan = {}
    total_weekly_calories = 0
    total_weekly_protein = 0
    global_ayurvedic_notes = set()
    global_used_ids = set()  # ← Zero repetition across all 7 days
    
    for day in range(1, 8):
        day_plan = {}
        
        # Residual tracking: start with exact daily targets
        remaining_cal = float(target_cal)
        remaining_pro = float(target_pro)
        remaining_carb = float(target_carb)
        remaining_fat = float(target_fat)
        
        remaining_ratio = 1.0  # sum of remaining meal ratios
        
        for meal_idx, (meal_name, slot_defs, base_ratio) in enumerate(MEAL_STRUCTURE):
            # Residual Correction: distribute remaining macros proportionally
            # across the remaining meals instead of using fixed percentages
            if remaining_ratio > 0:
                ratio_share = base_ratio / remaining_ratio
            else:
                ratio_share = base_ratio
            
            meal_target_cal = remaining_cal * ratio_share
            meal_target_pro = remaining_pro * ratio_share
            meal_target_carb = remaining_carb * ratio_share
            meal_target_fat = remaining_fat * ratio_share
            
            # Divide the meal's macro budget across its food slots
            n_slots = len(slot_defs)
            meal_items = []
            
            # ── PIN INJECTION: Force iconic regional dishes into this slot ──
            pin_key = (day, meal_name)
            if pin_key in pinned_foods:
                pinned = pinned_foods[pin_key]
                fid = pinned.get("id", pinned.get("name"))
                if fid not in global_used_ids:
                    meal_items.append(pinned)
                    global_used_ids.add(fid)
            
            slot_cal_remaining = meal_target_cal
            slot_pro_remaining = meal_target_pro
            slot_carb_remaining = meal_target_carb
            slot_fat_remaining = meal_target_fat
            
            # Subtract pinned food's contribution from slot budget
            for pinned_item in meal_items:
                slot_cal_remaining -= pinned_item.get("calories", 0)
                slot_pro_remaining -= pinned_item.get("protein", 0)
                slot_carb_remaining -= pinned_item.get("carbs", 0)
                slot_fat_remaining -= pinned_item.get("fats", 0)
            
            # How many slots were pre-filled by pins?
            pre_filled = len(meal_items)
            
            for slot_idx, pool_names in enumerate(slot_defs):
                # Skip slots that were filled by pinned foods
                if slot_idx < pre_filled:
                    continue
                slots_left = n_slots - slot_idx
                slot_target_cal = slot_cal_remaining / slots_left
                slot_target_pro = slot_pro_remaining / slots_left
                slot_target_carb = slot_carb_remaining / slots_left
                slot_target_fat = slot_fat_remaining / slots_left
                
                # Try each pool name in priority order
                best_food = None
                for pool_name in pool_names:
                    candidate = pick_best(
                        pools.get(pool_name, []),
                        slot_target_cal, slot_target_pro,
                        slot_target_carb, slot_target_fat,
                        global_used_ids,
                        day_seed=day,
                        exclude_heavy=(meal_name == "Dinner")
                    )
                    if candidate:
                        best_food = candidate
                        break
                
                if best_food:
                    meal_items.append(best_food)
                    fid = best_food.get("id", best_food.get("name"))
                    global_used_ids.add(fid)
                    
                    # Subtract what this food actually provides from the slot remainder
                    slot_cal_remaining -= best_food.get("calories", 0)
                    slot_pro_remaining -= best_food.get("protein", 0)
                    slot_carb_remaining -= best_food.get("carbs", 0)
                    slot_fat_remaining -= best_food.get("fats", 0)
            
            day_plan[meal_name] = meal_items
            
            # Subtract actual meal totals from the day's remaining budget
            actual_cal = sum(f.get("calories", 0) for f in meal_items)
            actual_pro = sum(f.get("protein", 0) for f in meal_items)
            actual_carb = sum(f.get("carbs", 0) for f in meal_items)
            actual_fat = sum(f.get("fats", 0) for f in meal_items)
            
            remaining_cal -= actual_cal
            remaining_pro -= actual_pro
            remaining_carb -= actual_carb
            remaining_fat -= actual_fat
            remaining_ratio -= base_ratio
        
        # ── Calculate actual daily totals ──
        daily_cal = sum(f.get("calories", 0) for items in day_plan.values() for f in items)
        daily_pro = sum(f.get("protein", 0) for items in day_plan.values() for f in items)
        daily_carb = sum(f.get("carbs", 0) for items in day_plan.values() for f in items)
        daily_fat = sum(f.get("fats", 0) for items in day_plan.values() for f in items)
        
        # Simplify output and calculate meal-level macros
        simplified_plan = {}
        for meal_name, items in day_plan.items():
            meal_cals = sum(item.get("calories", 0) for item in items)
            meal_pro = sum(item.get("protein", 0) for item in items)
            meal_carb = sum(item.get("carbs", 0) for item in items)
            meal_fat = sum(item.get("fats", 0) for item in items)
            
            simplified_plan[meal_name] = {
                "items": [item["name"] for item in items],
                "calories": round(meal_cals),
                "protein": round(meal_pro),
                "carbs": round(meal_carb),
                "fat": round(meal_fat)
            }
        
        weekly_plan[f"Day {day}"] = {
            "meals": simplified_plan,
            "daily_calories": round(daily_cal, 2),
            "daily_protein": round(daily_pro, 2),
            "daily_carbs": round(daily_carb, 2),
            "daily_fat": round(daily_fat, 2)
        }
        
        # Extract Ayurvedic notes for the LLM prompt
        for meal_items in day_plan.values():
            for item in meal_items:
                note = item.get("ayurveda", {}).get("notes", "")
                if note and note != "No specific Ayurvedic notes available.":
                    global_ayurvedic_notes.add(note)
        
        total_weekly_calories += daily_cal
        total_weekly_protein += daily_pro
        
        print(f"  Day {day}: {round(daily_cal)}kcal | {round(daily_pro)}g Pro | {round(daily_carb)}g Carb | {round(daily_fat)}g Fat")
    
    print(f"═══ Algorithm finished in {time.time() - start_time:.4f}s ═══")
    print(f"  Total unique foods used: {len(global_used_ids)}")
    
    # ── PHASE 4: LLM Doctor's Note ─────────────────────────────────
    llm_summary = "LLM formatting skipped (no API key)"
    
    sample_notes = list(global_ayurvedic_notes)[:4]
    notes_context = "\n".join([f"- {n}" for n in sample_notes])
    
    prompt = f"""Act as an Ayurvedic scholar and doctor. I have used a strict mathematical constraint solver to generate a 7-day {data.dosha} pacifying meal plan.
    
Here are the exact Ayurvedic properties of some ingredients the algorithm selected for this patient:
{notes_context}

Write a very brief 2-sentence encouraging opening message to the patient about this plan. Mention that the foods were specifically selected based on their Gunas (like Guru/Laghu etc) to pacify {data.dosha}. Do not list the meals, just the greeting."""
    
    if groq_client:
        try:
            groq_models = os.getenv("GROQ_MODELS", "mixtral-8x7b-32768")
            model_name = groq_models.split(",")[0] if "," in groq_models else groq_models
            
            completion = groq_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                timeout=5.0
            )
            llm_summary = completion.choices[0].message.content.strip()
        except Exception as e:
            llm_summary = f"Groq Error: {str(e)}"
    else:
        llm_summary = "No LLM API key configured. Set GROQ_API_KEY in .env"
            
    print(f"  LLM finished. Total time: {time.time() - start_time:.2f}s")
    return {
        "status": "success",
        "target_dosha": data.dosha,
        "cuisine_filter": data.cuisine,
        "llm_doctor_note": llm_summary,
        "weekly_plan": weekly_plan,
        "average_daily_calories": round(total_weekly_calories / 7, 2),
        "average_daily_protein": round(total_weekly_protein / 7, 2),
        "average_daily_carbs": round(sum(d["daily_carbs"] for d in weekly_plan.values()) / 7, 2),
        "average_daily_fat": round(sum(d["daily_fat"] for d in weekly_plan.values()) / 7, 2),
        "algorithm": "Multi-Objective Greedy Constraint Satisfaction v3.0 with Residual Correction & Diversity Enforcement",
        "total_unique_foods": len(global_used_ids),
        "message": f"Mathematically optimized 7-Day Plan for {data.dosha} constraints in {data.cuisine} cuisine from a database of {len(nutritional_db)} items."
    }

# ---------------------------------------------------------
# Pipeline C: RAG Chatbot v2.0
# Multi-Document Retrieval + Augmented Generation
# ---------------------------------------------------------
class ChatRequest(BaseModel):
    query: str

@app.post("/api/chat/")
def chat(data: ChatRequest):
    """
    ================================================================
    AyurNutri RAG Chatbot v2.0
    Algorithm: TF-IDF Cosine Similarity + Top-K Multi-Document 
               Retrieval + LLM Augmented Generation
    
    Pipeline:
    1. RETRIEVE: Vectorize user query, compute cosine similarity 
       against all corpus documents, return Top-K (K=3) matches
    2. RANK: Sort by relevance score, apply confidence threshold
    3. AUGMENT: Fuse multi-document context into a single prompt
    4. GENERATE: Send bounded prompt to LLM with strict system role
    5. RETURN: Response + retrieved sources + confidence scores
    ================================================================
    """
    import time
    start_time = time.time()
    
    try:
        # ---- Load Vector DB ----
        try:
            vector_db = joblib.load("vector_db.pkl")
            corpus = vector_db["corpus"]
            vectorizer = vector_db["vectorizer"]
            tfidf_matrix = vector_db["tfidf_matrix"]
        except Exception as e:
            return {"status": "error", "message": f"Vector DB not found. Run build_vector_db.py first. ({e})"}
        
        # ---- PHASE 1: RETRIEVE (Top-K Cosine Similarity) ----
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        query_vec = vectorizer.transform([data.query])
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        # Get Top-K indices sorted by descending similarity
        TOP_K = 3
        top_k_indices = np.argsort(similarities)[::-1][:TOP_K]
        
        # ---- PHASE 2: RANK + Confidence Threshold ----
        RELEVANCE_THRESHOLD = 0.05
        
        retrieved_docs = []
        for idx in top_k_indices:
            score = float(similarities[idx])
            if score >= RELEVANCE_THRESHOLD:
                retrieved_docs.append({
                    "topic": corpus[idx]["topic"],
                    "content": corpus[idx]["content"],
                    "score": round(score, 4),
                    "doc_id": corpus[idx].get("id", idx)
                })
        
        # ---- PHASE 3: AUGMENT (Build LLM Prompt) ----
        SYSTEM_PROMPT = """You are Dr. AyurVaidya, a highly qualified Ayurvedic physician (BAMS, MD Ayurveda) 
and scholar with 20 years of clinical experience. You specialize in Ayurvedic nutrition, 
Dosha assessment, and personalized diet planning.

STRICT RULES:
1. Answer ONLY from the Retrieved Ayurvedic Context provided below. Do NOT hallucinate or make up information.
2. If the context does not contain the answer, honestly say: "This topic is not covered in my current knowledge base."
3. Use proper Sanskrit Ayurvedic terminology (with English translations in parentheses) for credibility.
4. For ANY medical emergency or serious health condition, ALWAYS advise the user to consult a qualified doctor immediately.
5. Keep responses concise but scientifically grounded (3-5 sentences max).
6. If the user asks a non-Ayurvedic question, politely redirect them.
7. Never recommend specific dosages without advising physician consultation."""

        if not retrieved_docs:
            # No relevant documents found
            prompt = f"""{SYSTEM_PROMPT}

The user said: "{data.query}"

No relevant Ayurvedic context was found in the knowledge base for this query (similarity scores below threshold).
If the user is greeting you, respond warmly as Dr. AyurVaidya.
If the query is off-topic, politely explain that you are an Ayurvedic Nutrition specialist and ask how you can help with their Dosha balance, diet, or wellness."""
            
            retrieved_context_text = "No relevant context found (all scores below threshold)."
        else:
            # Fuse Top-K documents into a single rich context
            context_blocks = []
            for i, doc in enumerate(retrieved_docs, 1):
                context_blocks.append(
                    f"[Source {i} | Topic: {doc['topic']} | Relevance: {doc['score']:.1%}]\n{doc['content']}"
                )
            
            fused_context = "\n\n".join(context_blocks)
            retrieved_context_text = fused_context
            
            prompt = f"""{SYSTEM_PROMPT}

Retrieved Ayurvedic Context (Top-{len(retrieved_docs)} most relevant documents):
---
{fused_context}
---

User Question: {data.query}

Provide a clear, authoritative answer using ONLY the context above. Cite which source(s) you used."""
        
        # ---- PHASE 4: GENERATE (Neural NLP Engine) ----
        response_text = generate_neural_chat(data.query, retrieved_context_text)
        elapsed = round(time.time() - start_time, 3)
        
        # ---- PHASE 5: RETURN with full metadata ----
        return {
            "status": "success",
            "response": response_text,
            "retrieved_context": retrieved_context_text,
            "retrieval_metadata": {
                "top_k": TOP_K,
                "documents_retrieved": len(retrieved_docs),
                "sources": [{"topic": d["topic"], "relevance_score": d["score"]} for d in retrieved_docs],
                "corpus_size": len(corpus),
                "vocabulary_size": len(vectorizer.vocabulary_),
                "threshold": RELEVANCE_THRESHOLD
            },
            "algorithm": "TF-IDF Cosine Similarity + Top-3 Multi-Document RAG v2.0",
            "response_time_seconds": elapsed
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

class PromptRequest(BaseModel):
    prompt: str

@app.post("/api/generate-llm/")
def generate_llm(data: PromptRequest):
    """Generic endpoint to pass a prompt to Groq and return the JSON response."""
    if not groq_client:
        return {"status": "error", "message": "Groq API key is missing. Set GROQ_API_KEY in backend."}
    
    try:
        groq_models = os.getenv("GROQ_MODELS", "mixtral-8x7b-32768")
        model_name = groq_models.split(",")[0] if "," in groq_models else groq_models
        
        completion = groq_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant. Always return valid JSON when requested."},
                {"role": "user", "content": data.prompt}
            ],
            temperature=0.3,
            max_tokens=4000,
            timeout=15.0
        )
        
        response_text = completion.choices[0].message.content.strip()
        
        # In case the model wrapped it in markdown codeblocks:
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
            
        return {"status": "success", "response": response_text.strip()}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

class RecipeRequest(BaseModel):
    query: str
    dosha_type: str = None

@app.post("/api/generate-recipes/")
def generate_recipes(data: RecipeRequest):
    """Generate 3 authentic Ayurvedic recipes based on user input and dosha."""
    if not groq_client:
        return {"status": "error", "message": "Groq API key is missing. Set GROQ_API_KEY in backend."}
    
    try:
        # 1. Load the real database to fetch actual ingredients
        db_path = os.path.join(os.path.dirname(__file__), "nutritional_db.json")
        with open(db_path, "r", encoding="utf-8") as f:
            food_db = json.load(f)
            
        # 2. Constraint Algorithm: Filter foods based on user's Dosha
        valid_foods = []
        for food in food_db:
            dosha_effect = food.get("ayurveda", {}).get("dosha_effect", {})
            if data.dosha_type and dosha_effect.get(data.dosha_type) == "Pacifies":
                valid_foods.append(food["name"])
            elif not data.dosha_type:
                # If no dosha is provided, pick neutral/tridoshic foods (just add all)
                valid_foods.append(food["name"])
                
        # 3. Randomly select 3 optimized ingredients from the database
        import random
        selected_ingredients = random.sample(valid_foods, min(3, len(valid_foods)))
        
        # 4. Pass the real database ingredients to our Neural NLP Module for formatting
        response_text = generate_neural_recipe(data.query, data.dosha_type, selected_ingredients)
        return {"status": "success", "response": response_text.strip()}
    except Exception as e:
        return {"status": "error", "message": str(e)}

class VisionRequest(BaseModel):
    prompt: str
    base64_image: str
    mime_type: str = "image/jpeg"

@app.post("/api/analyze-food/")
def analyze_food(data: VisionRequest):
    """Analyze food images using Groq Vision models."""
    if not groq_client:
        return {"status": "error", "message": "Groq API key is missing. Set GROQ_API_KEY in backend."}
    
    try:
        completion = groq_client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": data.prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{data.mime_type};base64,{data.base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.7,
            max_tokens=4000,
            timeout=15.0
        )
        
        response_text = completion.choices[0].message.content.strip()
        
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
            
        return {"status": "success", "response": response_text.strip()}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}
