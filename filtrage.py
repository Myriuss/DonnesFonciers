import pandas as pd

# Fonction principale pour traiter les données
def traiter_donnees(fichier_input, fichier_output):
    df = pd.read_csv(fichier_input, delimiter='|', low_memory=False)
    print(f"Nombre de lignes avant traitement : {len(df)}")

    colonnes_a_supprimer = [
        'Reference document', '1 Articles CGI', '2 Articles CGI', '3 Articles CGI',
        '4 Articles CGI', '5 Articles CGI', 'No disposition', 'Prefixe de section',
        'Section', 'No plan', 'No Volume', '1er lot', 'Surface Carrez du 1er lot',
        '2eme lot', 'Surface Carrez du 2eme lot', '3eme lot', 'Surface Carrez du 3eme lot',
        '4eme lot', 'Surface Carrez du 4eme lot', '5eme lot', 'Surface Carrez du 5eme lot',
        'Nombre de lots', 'Identifiant local', 'Nature culture', 'Nature culture speciale',
        'Identifiant de document', 'Code postal', 'Code departement', 'Code voie', 'Type de voie',
        'No voie', 'B/T/Q', 'Code postal'
    ]
    df = df.drop(columns=colonnes_a_supprimer, errors='ignore')
    print(f"Nombre de lignes après suppression des colonnes non pertinentes : {len(df)}")

    # Filtrage pour la ville de Toulouse
    df['Commune'] = df['Commune'].str.upper().str.strip()
    df_toulouse = df[df['Commune'] == 'TOULOUSE']
    print(f"Nombre de lignes après filtrage pour Toulouse : {len(df_toulouse)}")

    # Conversion de la colonne Surface terrain en numérique
    # Utilisation de .loc pour éviter le SettingWithCopyWarning
    df_toulouse.loc[:, 'Surface terrain'] = pd.to_numeric(df_toulouse['Surface terrain'], errors='coerce')

    # Filtrage pour les surfaces et valeurs foncières valides
    df_toulouse = df_toulouse[(df_toulouse['Surface terrain'] > 0) & (df_toulouse['Valeur fonciere'].notnull())]
    print(f"Nombre de lignes après filtrage des surfaces et valeurs foncières : {len(df_toulouse)}")

    #  Sauvegarde du DataFrame filtré dans un fichier CSV
    df_toulouse.to_csv(fichier_output, index=False)
    print(f"Fichier filtré sauvegardé sous : {fichier_output}")

# Exécution du programme
if __name__ == "__main__":
    fichier_input = 'valeursfoncieres-2021.txt'
    fichier_output = 'toulouse_filtered_data.csv'
    traiter_donnees(fichier_input, fichier_output)
