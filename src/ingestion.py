import pandas as pd

def load_data():
    """
    Simulates fetching data from the 'mlsum' dataset.
    Returns a pandas DataFrame.
    """
    print("⏳ [Ingestion] Fetching raw data...")
    # In a real app, this would fetch from a database or API
    data = [
        {"text": "Die deutsche Wirtschaft wächst langsam. Experten sind vorsichtig optimistisch für 2026.", 
         "summary": "Wirtschaft wächst langsam."},
        {"text": "Der FC Bayern München hat gestern 3:0 gewonnen. Harry Kane erzielte zwei Tore.", 
         "summary": "Bayern gewinnt 3:0."}
    ]
    df = pd.DataFrame(data)
    print(f"✅ [Ingestion] Loaded {len(df)} samples.")
    return df

if __name__ == "__main__":
    load_data()