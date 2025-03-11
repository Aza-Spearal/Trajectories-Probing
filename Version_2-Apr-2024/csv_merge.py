import pandas as pd

# Lire les fichiers CSV dans des DataFrames
df1 = pd.read_csv('mazy_corpus_activation.csv')
df2 = pd.read_csv('2mazy_corpus_activation.csv')

# Fusionner les DataFrames en utilisant la colonne commune comme index
merged_df = pd.merge(df1, df2, on='Model')

# Trier le DataFrame fusionné selon l'index
merged_df.sort_values(by='Model', inplace=True)

# Réindexer le DataFrame final si nécessaire
merged_df.reset_index(drop=True, inplace=True)

# Enregistrer le DataFrame fusionné dans un nouveau fichier CSV
merged_df.to_csv('mazy_corpus_activation_final.csv', index=False)