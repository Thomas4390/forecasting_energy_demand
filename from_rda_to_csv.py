import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import pandas as pd
import os

# Activer la conversion automatique des objets R vers pandas
pandas2ri.activate()


def convert_rda_to_csv(rda_path: str, csv_path: str) -> None:
    """
    Convertit un fichier RDA en CSV.

    Parameters:
    rda_path (str): Le chemin du fichier .rda à convertir.
    csv_path (str): Le chemin du fichier de sortie .csv.
    """
    try:
        # Charger le fichier .rda
        robjects.r['load'](rda_path)

        # Obtenir le nom des objets chargés
        loaded_objects = robjects.r['ls']()
        if not loaded_objects:
            raise ValueError("Aucun objet trouvé dans le fichier .rda.")

        # Convertir le premier objet chargé en DataFrame pandas
        r_object = robjects.r[loaded_objects[0]]
        df: pd.DataFrame = pandas2ri.rpy2py(r_object)

        # Sauvegarder le DataFrame en CSV
        df.to_csv(csv_path, index=False)
        print(f"Le fichier CSV a été enregistré avec succès à : {csv_path}")

    except Exception as e:
        print(f"Une erreur s'est produite : {e}")


if __name__ == "__main__":
    # Chemin relatif vers le fichier .rda et le fichier .csv de sortie
    rda_file_path = "data/vic_elec.rda"
    csv_file_path = "data/vic_elec.csv"

    # Convertir le fichier .rda en .csv
    convert_rda_to_csv(rda_file_path, csv_file_path)
