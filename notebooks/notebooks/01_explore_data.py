from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]  # smartserve-ai root
DATA_PATH = BASE_DIR / "data" / "raw" / "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"

faq = pd.read_csv(DATA_PATH)
print(faq[['instruction', 'response']].head(5))
