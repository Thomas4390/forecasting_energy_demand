import pandas as pd
from plotly import graph_objects as go
import matplotlib.pyplot as plt
from typing import Tuple, Union
from astral.sun import sun
from astral import LocationInfo
import numpy as np


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


def plot_predictions_vs_real(data_test: pd.DataFrame,
                             predictions: pd.DataFrame,
                             demand_column: str = 'Demand',
                             prediction_column: str = 'pred',
                             title: str = "Real value vs predicted in test data",
                             figsize: Tuple[int, int] = (800, 400)) -> None:
    """
    Trace les prédictions par rapport aux valeurs réelles en utilisant plotly pour l'interactivité.

    Parameters:
    data_test (pd.DataFrame): DataFrame contenant les données de test réelles.
    predictions (pd.DataFrame): DataFrame contenant les données de prédiction.
    demand_column (str): Le nom de la colonne des demandes dans les données de test.
    prediction_column (str): Le nom de la colonne des prédictions dans les données de prédiction.
    title (str): Titre du graphique.
    figsize (Tuple[int, int]): Taille de la figure (width, height).
    """
    fig = go.Figure()

    # Tracé des données de test réelles
    fig.add_trace(go.Scatter(
        x=data_test.index,
        y=data_test[demand_column],
        mode='lines',
        name='Test',
        line=dict(color='blue', width=2)
    ))

    # Tracé des prédictions
    fig.add_trace(go.Scatter(
        x=predictions.index,
        y=predictions[prediction_column],
        mode='lines',
        name='Prediction',
        line=dict(color='red', width=2)
    ))

    # Mise à jour de la mise en page du graphique
    fig.update_layout(
        title=title,
        xaxis_title="Date time",
        yaxis_title="Demand",
        width=figsize[0],
        height=figsize[1],
        margin=dict(l=20, r=20, t=35, b=20),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.1,
            xanchor="left",
            x=0.001
        )
    )

    # Affichage du graphique
    fig.show()


def add_exogenous_features(data: pd.DataFrame,
                           name: str,
                           region: str,
                           timezone: str,
                           latitude: Union[float, str],
                           longitude: Union[float, str]) -> pd.DataFrame:
    """
    Ajoute des variables exogènes au DataFrame, y compris les caractéristiques du calendrier,
    la lumière du soleil, les jours fériés et les caractéristiques de la température.

    Parameters:
    data (pd.DataFrame): DataFrame contenant les données de base avec des colonnes de temps et de température.
    name (str): Nom de la localisation.
    region (str): Région de la localisation.
    timezone (str): Fuseau horaire de la localisation.
    latitude (Union[float, str]): Latitude de la localisation.
    longitude (Union[float, str]): Longitude de la localisation.

    Returns:
    pd.DataFrame: DataFrame contenant les caractéristiques exogènes ajoutées.
    """

    # Caractéristiques du calendrier
    calendar_features = pd.DataFrame(index=data.index)
    calendar_features['month'] = data.index.month
    calendar_features['week_of_year'] = data.index.isocalendar().week
    calendar_features['week_day'] = data.index.dayofweek + 1
    calendar_features['hour_day'] = data.index.hour + 1

    # Caractéristiques de la lumière du soleil avec valeurs fixes pour sunrise et sunset
    sun_light_features = pd.DataFrame(index=data.index)
    sun_light_features['sunrise_hour'] = 6
    sun_light_features['sunset_hour'] = 20
    sun_light_features['daylight_hours'] = sun_light_features['sunset_hour'] - sun_light_features['sunrise_hour']
    sun_light_features['is_daylight'] = np.where(
        (data.index.hour >= sun_light_features['sunrise_hour']) &
        (data.index.hour < sun_light_features['sunset_hour']),
        1,
        0
    )

    # Caractéristiques des jours fériés
    holiday_features = data[['Holiday']].astype(int)
    holiday_features['holiday_previous_day'] = holiday_features['Holiday'].shift(24, fill_value=0)
    holiday_features['holiday_next_day'] = holiday_features['Holiday'].shift(-24, fill_value=0)

    # Caractéristiques de la température
    temp_features = data[['Temperature']].copy()
    temp_features['temp_roll_mean_1_day'] = temp_features['Temperature'].rolling(24, min_periods=1).mean()
    temp_features['temp_roll_mean_7_day'] = temp_features['Temperature'].rolling(24 * 7, min_periods=1).mean()
    temp_features['temp_roll_max_1_day'] = temp_features['Temperature'].rolling(24, min_periods=1).max()
    temp_features['temp_roll_min_1_day'] = temp_features['Temperature'].rolling(24, min_periods=1).min()
    temp_features['temp_roll_max_7_day'] = temp_features['Temperature'].rolling(24 * 7, min_periods=1).max()
    temp_features['temp_roll_min_7_day'] = temp_features['Temperature'].rolling(24 * 7, min_periods=1).min()

    # Fusion de toutes les caractéristiques exogènes
    exogenous_features = pd.concat([
        calendar_features,
        sun_light_features,
        temp_features,
        holiday_features
    ], axis=1)

    return exogenous_features


def cyclical_encoding(data: pd.Series, cycle_length: int) -> pd.DataFrame:
    """
    Encode a cyclical feature with two new features sine and cosine.
    The minimum value of the feature is assumed to be 0. The maximum value
    of the feature is passed as an argument.

    Parameters
    ----------
    data : pd.Series
        Series with the feature to encode.
    cycle_length : int
        The length of the cycle. For example, 12 for months, 24 for hours, etc.
        This value is used to calculate the angle of the sin and cos.

    Returns
    -------
    result : pd.DataFrame
        Dataframe with the two new features sin and cos.

    """

    sin = np.sin(2 * np.pi * data / cycle_length)
    cos = np.cos(2 * np.pi * data / cycle_length)
    result = pd.DataFrame({
        f"{data.name}_sin": sin,
        f"{data.name}_cos": cos
    })

    return result


def plot_prediction_intervals_vs_real(predictions: pd.DataFrame,
                                      data_test: pd.DataFrame,
                                      pred_column: str = 'pred',
                                      demand_column: str = 'Demand',
                                      upper_bound_column: str = 'upper_bound',
                                      lower_bound_column: str = 'lower_bound',
                                      title: str = "Real value vs predicted in test data",
                                      figsize: Tuple[int, int] = (800, 400)) -> None:
    """
    Trace les intervalles de prédiction par rapport aux valeurs réelles en utilisant plotly.

    Parameters:
    predictions (pd.DataFrame): DataFrame contenant les prédictions et les intervalles de confiance.
    data_test (pd.DataFrame): DataFrame contenant les valeurs réelles de test.
    pred_column (str): Le nom de la colonne des prédictions dans le DataFrame des prédictions.
    demand_column (str): Le nom de la colonne des valeurs réelles dans le DataFrame de test.
    upper_bound_column (str): Le nom de la colonne de la borne supérieure dans le DataFrame des prédictions.
    lower_bound_column (str): Le nom de la colonne de la borne inférieure dans le DataFrame des prédictions.
    title (str): Titre du graphique.
    figsize (Tuple[int, int]): Taille de la figure (width, height).
    """

    fig = go.Figure([
        go.Scatter(
            name='Prediction',
            x=predictions.index,
            y=predictions[pred_column],
            mode='lines',
        ),
        go.Scatter(
            name='Real value',
            x=data_test.index,
            y=data_test[demand_column],
            mode='lines',
        ),
        go.Scatter(
            name='Upper Bound',
            x=predictions.index,
            y=predictions[upper_bound_column],
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='Lower Bound',
            x=predictions.index,
            y=predictions[lower_bound_column],
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        )
    ])

    fig.update_layout(
        title=title,
        xaxis_title="Date time",
        yaxis_title="Demand",
        width=figsize[0],
        height=figsize[1],
        margin=dict(l=20, r=20, t=35, b=20),
        hovermode="x",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.1,
            xanchor="left",
            x=0.001
        )
    )

    fig.show()




