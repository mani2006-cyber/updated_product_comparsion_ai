import pandas as pd
general = pd.read_csv('data/catalog_relationship_pairs.csv')
general_filtered = general[general['relationship_label'] != 'WEAKLY_SIMILAR']  # drop the noisy class entirely

audio = pd.read_csv('data/relationship_pairs_with_nearmiss_v2.csv')
combined = pd.concat([audio, general_filtered], ignore_index=True)
combined.to_csv('data/relationship_pairs_general_v2.csv', index=False)
print(combined['relationship_label'].value_counts())