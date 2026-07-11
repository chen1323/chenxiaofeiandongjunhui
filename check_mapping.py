import os
import pandas as pd
import re

def clean_name(folder_name):
    name = re.sub(r'^\d+\s*-\s*', '', folder_name)
    if ',' in name:
        parts = name.split(',', 1)
        name = f"{parts[1].strip()} {parts[0].strip()}"
    return name.title()

df = pd.read_csv('20260509_2015-2018_rookie_dataset.csv', index_col=0)

base_dirs = {
    2015: "2015-2018_rookie_raw_cv_and_jmp/2015",
    2016: "2015-2018_rookie_raw_cv_and_jmp/2016",
    2017: "2015-2018_rookie_raw_cv_and_jmp/2017"
}

all_mappings = []
start_id = 1

for year in [2015, 2016, 2017]:
    # rows in CSV for this year
    num_rows = len(df[df['year'] == year])
    
    # folders for this year
    folders = [f for f in os.scandir(base_dirs[year]) if f.is_dir()]
    sorted_folders = sorted(folders, key=lambda f: f.name)
    
    print(f"Year {year}: CSV rows = {num_rows}, Folders = {len(sorted_folders)}")
    
    for i, folder in enumerate(sorted_folders):
        csv_id = start_id + i
        all_mappings.append({
            "csv_candidate_id": csv_id,
            "year": year,
            "folder_name": folder.name,
            "candidate_name": clean_name(folder.name)
        })
    start_id += num_rows

mapping_df = pd.DataFrame(all_mappings)
mapping_df.to_csv("candidate_mapping_2015_2017.csv", index=False)
print("Saved to candidate_mapping_2015_2017.csv")
print(mapping_df.head(10))
