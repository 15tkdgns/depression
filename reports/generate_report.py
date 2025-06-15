import pandas as pd

def save_report(data_dict, filename):
    with pd.ExcelWriter(filename) as writer:
        for sheet_name, df in data_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
