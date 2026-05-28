import torch
import numpy as np
import random
import json
import os
from datetime import datetime
from torch.utils.data import DataLoader
from src.data.preprocess import load_config, load_batadal, load_and_concat_skab
from src.data.split import split_and_scale_batadal, get_skab_splits
from src.models.deep_learning import TimeSeriesDataset, LSTMModel, CNN1DModel, train_model
from src.models.automata import ProbabilisticAutomata
from src.utils.metrics import calculate_metrics

def set_seed(seed):
    """Her deneme için rastgelelikleri sabitler (Reproducibility)"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    config = load_config()

    # Verileri Yükle
    batadal_df = load_batadal(config['data']['batadal_path'])
    
    # Not: İleride SKAB için Cross-Dataset yapılacağı zaman bu kısmı kullanacağız.
    # skab_df = load_and_concat_skab(config['data']['skab_dir']) 

    # Veriyi Böl ve Ölçeklendir (Sızıntı Kurallarına Uygun)
    b_train, b_val, b_test, b_feats, b_target = split_and_scale_batadal(batadal_df)

    # Parametreler
    window_size = config['params']['window_size']
    alphabet_size = config['params']['alphabet_size']
    batch_size = config['params']['batch_size']
    max_epochs = config['params']['max_epochs']
    patience = config['params']['early_stopping']
    random_seeds = config.get('random_seeds', [42, 123, 2026, 7, 999])

    # PyTorch Dataset Oluşturma
    train_dataset = TimeSeriesDataset(b_train[b_feats].values, b_train[b_target].values, window_size)
    val_dataset = TimeSeriesDataset(b_val[b_feats].values, b_val[b_target].values, window_size)
    test_dataset = TimeSeriesDataset(b_test[b_feats].values, b_test[b_target].values, window_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = len(b_feats)

    models_to_run = ["LSTM", "1D-CNN"]
    results = {m: {'acc': [], 'prec': [], 'rec': [], 'f1': []} for m in models_to_run}

    print("\n" + "="*50)
    print("DERIN OGRENME MODELLERI (LSTM & 1D-CNN) EGITIMI")
    print("="*50)

    # 1. GÖREV: 5 Farklı Random Seed ve 2 Farklı Model Döngüsü
    for seed in random_seeds:
        set_seed(seed)
        print(f"\n>>> SEED: {seed} <<<")
        
        for model_name in models_to_run:
            if model_name == "LSTM":
                model = LSTMModel(input_size=input_size)
            else:
                model = CNN1DModel(input_size=input_size, window_size=window_size)
            
            # Eğitimi Başlat
            trained_model = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=max_epochs,
                patience=patience,
                device=device
            )
            
            # Test Seti Üzerinde Değerlendirme (Inference)
            trained_model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(device)
                    outputs = trained_model(X_batch)
                    preds = (outputs.cpu().numpy() > 0.5).astype(int)
                    all_preds.extend(preds)
                    all_targets.extend(y_batch.numpy())
            
            # Metrikleri Hesapla
            acc, prec, rec, f1 = calculate_metrics(all_targets, all_preds)
            print(f"[{model_name}] Test F1: {f1:.4f} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")
            
            # Sonuçları Kaydet
            results[model_name]['acc'].append(acc)
            results[model_name]['prec'].append(prec)
            results[model_name]['rec'].append(rec)
            results[model_name]['f1'].append(f1)

    print("\n" + "="*50)
    print("5 SEED ICIN ORTALAMA VE STANDART SAPMA SONUCLARI")
    print("="*50)
    for model_name in models_to_run:
        f1_mean = np.mean(results[model_name]['f1'])
        f1_std = np.std(results[model_name]['f1'])
        print(f"{model_name:8s} -> Ortalama F1-Score: {f1_mean:.4f} ± {f1_std:.4f}")
    print("="*50)

    # Sonuclari Kalici Olarak JSON Dosyasina Kaydetme
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_data = {
        "timestamp": timestamp,
        "config_params": config['params'],
        "random_seeds": random_seeds,
        "results": results,
        "summary": {}
    }
    
    for model_name in models_to_run:
        log_data["summary"][model_name] = {
            "f1_mean": float(np.mean(results[model_name]['f1'])),
            "f1_std": float(np.std(results[model_name]['f1']))
        }
        
    log_file_path = f"logs/experiment_results_{timestamp}.json"
    with open(log_file_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=4)
        
    print(f"\n[BILGI] Tüm deney parametreleri ve sonuclar basariyla kaydedildi: {log_file_path}")

    # -----------------------------------------------------------
    # OTOMATA KISMI (Geçici Olarak Eskisi Gibi Bırakıldı)
    # -----------------------------------------------------------
    print("\n--- OLASILIKSAL OTOMATA TESTI (Temel) ---")
    automata = ProbabilisticAutomata(window_size=window_size, alphabet_size=alphabet_size)
    
    train_ts = b_train['PC1'].values
    print("Otomata modeli egitiliyor...")
    automata.fit(train_ts)
    print("Otomata egitimi tamamlandi.")

    test_ts = b_test['PC1'].values
    test_patterns = automata._extract_patterns(test_ts)
    
    print("\nIlk test adimi icin aciklama ciktisi:")
    if len(test_patterns) > 1:
        prev_state = test_patterns[0]
        incoming_pattern = test_patterns[1]
        explanation = automata.explain_step(prev_state, incoming_pattern, time_step=1)
        print(json.dumps(explanation, indent=4))

if __name__ == "__main__":
    main()