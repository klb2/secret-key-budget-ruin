import pandas as pd


def export_results(results, filename):
    df = pd.DataFrame.from_dict(results)
    df.to_csv(filename, sep='\t', index=False)
