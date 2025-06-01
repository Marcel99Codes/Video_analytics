import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

print("Start")

data_path = './csv2'
csv_files = sorted(glob.glob(os.path.join(data_path, '*.csv')))

output_path = './output'
os.makedirs(output_path, exist_ok=True)

for file_path in csv_files:
    df = pd.read_csv(file_path)

    if not {'Step', 'Value'}.issubset(df.columns):
        df.columns = ["Wall time", "Step", "Value"]

    filename = os.path.splitext(os.path.basename(file_path))[0]

    plt.figure(figsize=(8, 5))
    plt.plot(df['Step'], df['Value'], marker='o', linestyle='-', color='blue')
    plt.title(f"{filename} over 15 epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.grid(True)
    plt.tight_layout()

    output_file = os.path.join(output_path, f"{filename}.png")
    plt.savefig(output_file)
    plt.close()
