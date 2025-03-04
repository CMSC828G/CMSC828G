""" A helper script to analyze the NVTX times from the report by `nsys recipe nvtx_gpu_proj_trace ...`
"""
from argparse import ArgumentParser
import pandas as pd

parser = ArgumentParser(description="Analyze NVTX times from a report.")
parser.add_argument("report", type=str, help="Path to parquet report file.")
parser.add_argument("output", type=str, help="Path to output CSV file.")
args = parser.parse_args()

df = pd.read_parquet(args.report)
stats = df.groupby("Name").agg({"Projected Duration": ["min", "max", "mean", "std", "median"]})
stats = stats / 1e9  # Convert to seconds

print(stats)
stats.to_csv(args.output)