import os
import numpy as np
import csv
from pyproj import Proj, transform

# Set up UTM to WGS84 projection
utm_proj = Proj(proj="utm", zone=54, south=True, ellps="WGS84")
wgs_proj = Proj(proj="latlong", datum="WGS84")

# Dataset parameters
resolution = 10  # meters per pixel
half_extent = 256 * resolution  # Half-grid extent (in meters)

# Input and output directories
input_dir = "/Users/wiebkezink/Documents/Uni Münster/MA/dataset"  # Update with your directory
output_dir = "/Users/wiebkezink/Documents/Uni Münster/MA/dataset"  # Update with your directory

# Thresholds for gX values
thresholds = [5, 10, 15, 20, 25, 30]

# Collect data for CSV
data_entries = []

# Iterate through all .npz files
for file_name in os.listdir(input_dir):
    if file_name.endswith(".npz"):
        file_path = os.path.join(input_dir, file_name)
        npz_file = np.load(file_path)
        
        # Extract data
        labels = npz_file["labels"]  # Assuming `labels` is a 3D array
        
        # Compute gX values for `labels[0]`
        g_values = {f"g{t}": np.mean(labels[0] > t) for t in thresholds}
        
        # Compute totals
        totals = labels[0].size
        
        # Get UTM coordinates (from file metadata or assumed center)
        utm_x, utm_y = 300000, 5800000  # Replace with metadata or set default
        
        # Convert UTM to WGS84
        center_lon, center_lat = transform(utm_proj, wgs_proj, utm_x, utm_y)
        
        # Prepare entry
        entry = {
            "paths": file_path,
            **g_values,
            "totals": totals,
            "longitudes": center_lon,
            "latitudes": center_lat,
        }
        data_entries.append(entry)

# Write CSV files
csv_headers = ["paths"] + [f"g{t}" for t in thresholds] + ["totals", "longitudes", "latitudes"]

# Divide into test, val, and fixval (customize this logic if needed)
test_split = data_entries[:len(data_entries) // 3]
val_split = data_entries[len(data_entries) // 3 : 2 * len(data_entries) // 3]
fixval_split = data_entries[2 * len(data_entries) // 3 :]

for split_name, split_data in zip(["test", "val", "fixval"], [test_split, val_split, fixval_split]):
    output_file = os.path.join(output_dir, f"{split_name}.csv")
    with open(output_file, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(split_data)

print("CSV files created successfully!")
