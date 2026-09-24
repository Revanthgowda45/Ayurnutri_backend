import pandas as pd
import random

def generate_synthetic_data(num_samples=1000):
    # We have 15 features matching the React Native frontend:
    # 1. body_frame
    # 2. walk_and_talk
    # 3. weather_reaction
    # 4. sweating
    # 5. appetite
    # 6. skin
    # 7. hair
    # 8. lips_and_teeth
    # 9. eyes
    # 10. general_signs
    # 11. memory
    # 12. mind
    # 13. mind_on_actions
    # 14. sleep_quality
    # 15. emotional_nature

    options = {
        'vata': 'vata',
        'pitta': 'pitta',
        'kapha': 'kapha'
    }
    
    data = []
    for _ in range(num_samples):
        # We will create a slight correlation to make the ML model learn properly
        dosha = random.choice(['Vata', 'Pitta', 'Kapha'])
        
        # Helper to pick answers based on the target dosha to create strong correlation
        def pick(target):
            if target == 'Vata':
                return random.choices(['vata', 'pitta', 'kapha'], weights=[0.8, 0.1, 0.1])[0]
            elif target == 'Pitta':
                return random.choices(['vata', 'pitta', 'kapha'], weights=[0.1, 0.8, 0.1])[0]
            else: # Kapha
                return random.choices(['vata', 'pitta', 'kapha'], weights=[0.1, 0.1, 0.8])[0]
            
        data.append({
            'body_frame': pick(dosha),
            'walk_and_talk': pick(dosha),
            'weather_reaction': pick(dosha),
            'sweating': pick(dosha),
            'appetite': pick(dosha),
            'skin': pick(dosha),
            'hair': pick(dosha),
            'lips_and_teeth': pick(dosha),
            'eyes': pick(dosha),
            'general_signs': pick(dosha),
            'memory': pick(dosha),
            'mind': pick(dosha),
            'mind_on_actions': pick(dosha),
            'sleep_quality': pick(dosha),
            'emotional_nature': pick(dosha),
            'target_dosha': dosha
        })
        
    df = pd.DataFrame(data)
    df.to_csv('dosha_data.csv', index=False)
    print(f"Generated {num_samples} records with 15 features and saved to dosha_data.csv")

if __name__ == "__main__":
    generate_synthetic_data()
