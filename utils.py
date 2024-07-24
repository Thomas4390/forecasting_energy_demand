import pandas as pd
from plotly import graph_objects as go
import matplotlib.pyplot as plt
from typing import Tuple


def split_data(df: pd.DataFrame,
               train_end: str,
               validation_end: str,
               start_date: str = '2012-01-01 00:00:00',
               end_date: str = '2014-12-30 23:00:00') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divise les données en ensembles d'entraînement, de validation et de test.

    Parameters:
    df (pd.DataFrame): DataFrame contenant les données complètes.
    train_end (str): Date de fin de l'ensemble d'entraînement.
    validation_end (str): Date de fin de l'ensemble de validation.
    start_date (str): Date de début de la période à considérer.
    end_date (str): Date de fin de la période à considérer.

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: Les ensembles de données, d'entraînement, de validation et de test.
    """
    # Filtrer les données pour la période spécifiée
    data = df.loc[start_date:end_date].copy()

    # Diviser les données en ensembles d'entraînement, de validation et de test
    data_train = data.loc[:train_end].copy()
    data_val = data.loc[train_end:validation_end].copy()
    data_test = data.loc[validation_end:].copy()

    # Afficher les informations sur chaque ensemble
    print(f"Train dates      : {data_train.index.min()} --- {data_train.index.max()}  (n={len(data_train)})")
    print(f"Validation dates : {data_val.index.min()} --- {data_val.index.max()}  (n={len(data_val)})")
    print(f"Test dates       : {data_test.index.min()} --- {data_test.index.max()}  (n={len(data_test)})")

    return data, data_train, data_val, data_test


def plot_interactive_time_series(data_train: pd.DataFrame, data_val: pd.DataFrame,
                                 data_test: pd.DataFrame,
                                 demand_column: str = 'Demand',
                                 title: str = 'Hourly energy demand',
                                 figsize: Tuple[int, int] = (850, 400)) -> None:
    """
    Trace une série temporelle interactive avec des partitions train, validation et test.

    Parameters:
    data_train (pd.DataFrame): DataFrame contenant les données d'entraînement.
    data_val (pd.DataFrame): DataFrame contenant les données de validation.
    data_test (pd.DataFrame): DataFrame contenant les données de test.
    demand_column (str): Le nom de la colonne des demandes dans les DataFrames.
    title (str): Titre du graphique.
    figsize (Tuple[int, int]): Taille de la figure (width, height).
    """
    fig = go.Figure()

    # Ajout des traces pour chaque partition
    fig.add_trace(go.Scatter(x=data_train.index, y=data_train[demand_column],
                             mode='lines', name='Train'))
    fig.add_trace(go.Scatter(x=data_val.index, y=data_val[demand_column],
                             mode='lines', name='Validation'))
    fig.add_trace(go.Scatter(x=data_test.index, y=data_test[demand_column],
                             mode='lines', name='Test'))

    # Mise à jour de la mise en page du graphique
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Demand",
        legend_title="Partition:",
        width=figsize[0],
        height=figsize[1],
        margin=dict(l=20, r=20, t=35, b=20),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1,
            xanchor="left",
            x=0.001
        )
    )

    # Affichage du graphique
    fig.show()


def plot_zoomed_time_series(data: pd.DataFrame, zoom_range: Tuple[str, str],
                            demand_column: str = 'Demand',
                            figsize: Tuple[int, int] = (8, 4)) -> None:
    """
    Trace une série temporelle avec une section zoomée.

    Parameters:
    data (pd.DataFrame): DataFrame contenant les données de la série temporelle.
    zoom_range (Tuple[str, str]): Période de temps à zoomer (format 'YYYY-MM-DD HH:MM:SS').
    demand_column (str): Le nom de la colonne des demandes dans le DataFrame.
    figsize (Tuple[int, int]): Taille de la figure.
    """
    fig = plt.figure(figsize=figsize)
    grid = plt.GridSpec(nrows=8, ncols=1, hspace=0.6, wspace=0)

    # Trace principale
    main_ax = fig.add_subplot(grid[:3, :])
    data[demand_column].plot(ax=main_ax, c='black', alpha=0.5, linewidth=0.5)

    min_y = data[demand_column].min()
    max_y = data[demand_column].max()

    main_ax.fill_between(zoom_range, min_y, max_y, facecolor='blue', alpha=0.5, zorder=0)
    main_ax.set_title(f'Electricity demand: {data.index.min()}, {data.index.max()}', fontsize=10)
    main_ax.set_xlabel('')

    # Trace zoomée
    zoom_ax = fig.add_subplot(grid[5:, :])
    data.loc[zoom_range[0]: zoom_range[1], demand_column].plot(ax=zoom_ax, color='blue', linewidth=1)
    zoom_ax.set_title(f'Electricity demand: {zoom_range}', fontsize=10)
    zoom_ax.set_xlabel('')

    plt.subplots_adjust(hspace=1)
    plt.show()




