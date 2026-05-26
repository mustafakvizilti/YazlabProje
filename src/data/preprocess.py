import pandas as pd
import numpy as np
import yaml
import os
import glob

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def load_batadal(data_path):
    print(f"BATADAL verisi yükleniyor: {data_path}")
    df = pd.read_csv(data_path)
    return df

def load_and_concat_skab(skab_dir):
    print(f"SKAB verileri birleştiriliyor: {skab_dir}")
    all_data = []
    
    for valve_folder in ['valve1', 'valve2']:
        folder_path = os.path.join(skab_dir, valve_folder)
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        
        for file in csv_files:
            df = pd.read_csv(file, sep=';') 
            
            df['source_group'] = valve_folder
            df['source_file'] = os.path.basename(file)
            
            all_data.append(df)
            
    skab_df = pd.concat(all_data, ignore_index=True)
    print(f"SKAB birleştirme tamamlandı. Toplam satır: {len(skab_df)}")
    return skab_df

if __name__ == "__main__":
    config = load_config()
    
    batadal_df = load_batadal(config['data']['batadal_path'])
    skab_df = load_and_concat_skab(config['data']['skab_dir'])
    
    print("\nVeriler başarıyla yüklendi")