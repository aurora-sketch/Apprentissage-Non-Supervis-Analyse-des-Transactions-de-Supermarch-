import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

# Importation du dataset dans un dataframe
df = pd.read_csv('Groceries_dataset.csv')

# Regroupement des articles par transaction (Member_number)
transactions = df.groupby('Member_number')['itemDescription'].apply(list)

# Transformation des transactions en un format binaire (one-hot encoding)
def encode_transactions(transactions):
    unique_items = list(set(item for transaction in transactions for item in transaction))
    encoded_df = pd.DataFrame(0, index=transactions.index, columns=unique_items)
    for idx, items_bought in transactions.items():
        for item in items_bought:
            encoded_df.at[idx, item] = 1
    return encoded_df

encoded_df = encode_transactions(transactions)

# Affichage des transactions binarisées
print("\nTransactions binarisées :")
print(encoded_df)

# Affichage des fréquences des articles sous forme d'histogramme
item_frequencies = encoded_df.sum().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
plt.bar(item_frequencies.index, item_frequencies.values, color='b')
plt.title("Fréquence des articles dans les transactions")
plt.xlabel("Articles")
plt.ylabel("Nombre d'apparitions")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Vérification de la densité des transactions binarisées
print("\nDensité des transactions binarisées :")
print(encoded_df.mean(axis=1))

# Garder seulement les articles qui apparaissent plus de X fois
min_freq = 5  # Par exemple, garder les articles apparaissant au moins 5 fois
item_frequencies = encoded_df.sum()
frequent_items = item_frequencies[item_frequencies >= min_freq].index

# Filtrer le DataFrame encodé pour ne garder que ces articles
encoded_df_filtered = encoded_df[frequent_items]

# Vérifier les transactions filtrées
print("\nTransactions filtrées :")
print(encoded_df_filtered)

# Application de l'algorithme FP-Growth avec un support minimum de 1%
frequent_itemsets_fpgrowth = fpgrowth(encoded_df_filtered, min_support=0.01, use_colnames=True)

# Vérification des itemsets fréquents avec FP-Growth
print("\nItemsets fréquents avec FP-Growth :")
print(frequent_itemsets_fpgrowth)

# Génération des règles d'association avec un seuil de confiance et de lift
rules = association_rules(frequent_itemsets_fpgrowth, metric="lift", min_threshold=1.5)

# Filtrage des règles qui ont un support et une confiance intéressants
rules_filtered = rules[(rules['confidence'] >= 0.7) & (rules['lift'] > 1)]

# Affichage des règles filtrées
print("\nRègles d'association filtrées :")
print(rules_filtered[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

def afficher_métriques(rules):
    """
    Affiche les métriques de Support, Confiance et Lift pour chaque règle d'association.
    """
    print("\nMétriques des règles d'association :")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

    # Affichage supplémentaire de métriques sous forme de résumé
    print("\nRésumé des métriques :")
    print(f"Support moyen : {rules['support'].mean():.4f}")
    print(f"Confiance moyenne : {rules['confidence'].mean():.4f}")
    print(f"Lift moyen : {rules['lift'].mean():.4f}")

# Affichage des règles filtrées
afficher_métriques(rules_filtered)
