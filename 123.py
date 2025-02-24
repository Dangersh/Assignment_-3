import pandas as pd

jurisdictions = ["AB", "BC", "NS", "ON", "PEI", "QC", "SK"]
data = []

for jurisdiction in jurisdictions:
        file = jurisdiction + ".CPI.1810000401 copy.csv"
        df = pd.read_csv(file)

for df in dfs 
        data.append(df)