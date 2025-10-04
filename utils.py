from datetime import datetime
import logging
import json
from pathlib import Path
from typing import Union
import pandas as pd


LOG_LEVEL = logging.INFO
def get_logger(s):
    logging.basicConfig(format='%(asctime)s|%(levelname)s|%(name)s (%(lineno)d): %(message)s',
                        datefmt="%d/%m/%y %H:%M:%S")
    logger = logging.getLogger(s)
    logger.setLevel(LOG_LEVEL)
    return logger

DATE = 'date'
DATETIME = 'datetime'
def load_config(config_path: Union[Path, str] = Path("config.json"), output_dir_suffix: str = None, add_date: str = DATE) -> dict:
    with open(config_path, encoding='utf-8') as f:
        config = json.load(f)
    output_dir_today = Path(config['output_dir'])
    if output_dir_suffix is not None:
        output_dir_today /= output_dir_suffix
    if add_date == DATE:
        output_dir_today /= datetime.now().date().isoformat()
    elif add_date == DATETIME:
        output_dir_today /= datetime.now().isoformat("_", 'minutes').replace(":", "-")
    config['output_dir'] = output_dir_today
    output_dir_today.mkdir(parents=True, exist_ok=True)
    for x in config.keys():
        if x.endswith('_dir'):
            config[x] = Path(config[x])
    return config


def prompt_config(config_def):
    import easygui_qt
    config = {}
    for field, default_response in config_def.items():
        t = '\t'*int((len(default_response)/4))
        config[field] = easygui_qt.get_string(message=f"Please enter {field}{t}", 
                                             default_response=default_response)
    return config


def replace_row_with_dataframe(df: pd.DataFrame, row_idx: int, replacement_df: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    """
    Replace a single row in a DataFrame with multiple rows from another DataFrame.
    
    Args:
        df: Original DataFrame
        row_idx: Index of the row to replace
        replacement_df: DataFrame containing replacement rows
        row: Original row data for filling missing columns
        
    Returns:
        DataFrame with the row replaced by the replacement rows
    """
    original_columns = df.columns.tolist()
    
    # Drop columns not present in the original, and add missing ones filled from the source row
    replacement_aligned = replacement_df.copy()
    # If replacement came as Series-stacked columns (some returns use .T), coerce to DataFrame
    if isinstance(replacement_aligned, pd.Series):
        replacement_aligned = replacement_aligned.to_frame().T

    for col in list(replacement_aligned.columns):
        if col not in original_columns:
            replacement_aligned.drop(columns=[col], inplace=True)

    for col in original_columns:
        if col not in replacement_aligned.columns:
            replacement_aligned[col] = row.get(col, None)

    # Reorder columns to match original
    replacement_aligned = replacement_aligned[original_columns]

    # Splice: rows before, replacement rows, rows after
    before = df.iloc[:row_idx]
    after = df.iloc[row_idx+1:]
    return pd.concat([before, replacement_aligned, after], ignore_index=True)