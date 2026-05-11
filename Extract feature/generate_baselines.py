import pandas as pd
import json
import random
from core.feature_extractor.extractor import CppFeatureExtractorV8
from tqdm import tqdm

def main():
    print("Loading jsonl...")
    with open("data/raw/train_data_10_model_AI.jsonl", "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    ai_lines = [l for l in lines if '"label": "AI"' in l]
    human_lines = [l for l in lines if '"label": "HUMAN"' in l]
    
    # Lấy mẫu ngẫu nhiên để xử lý nhanh (VD: 500 AI, 500 Human)
    random.seed(42)
    ai_sample = random.sample(ai_lines, min(500, len(ai_lines)))
    human_sample = random.sample(human_lines, min(500, len(human_lines)))
    
    extractor = CppFeatureExtractorV8()
    
    ai_features = []
    print("Processing AI samples...")
    for line in tqdm(ai_sample):
        data = json.loads(line)
        features = extractor.extract(data["code"])
        ai_features.append(features)
        
    human_features = []
    print("Processing Human samples...")
    for line in tqdm(human_sample):
        data = json.loads(line)
        features = extractor.extract(data["code"])
        human_features.append(features)
        
    ai_df = pd.DataFrame(ai_features)
    human_df = pd.DataFrame(human_features)
    
    baselines = {}
    for col in ai_df.columns:
        baselines[col] = {
            "ai": float(ai_df[col].mean()),
            "human": float(human_df[col].mean())
        }
        
    with open("models/zoo/baselines.json", "w", encoding="utf-8") as f:
        json.dump(baselines, f, indent=4)
        
    print("Saved baselines to models/zoo/baselines.json")

if __name__ == "__main__":
    main()
