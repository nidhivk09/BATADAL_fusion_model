import pandas as pd

df = pd.read_csv("/Users/nidhikulkarni/ics-anomaly-detection/data/BATADAL/test_whitebox_attack.csv")

count = (df["ATT_FLAG"] == 1).sum()

print("anomaly count is:", count)
