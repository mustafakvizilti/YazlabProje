import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from torch.utils.data import DataLoader

from src.data.preprocess import load_config, load_batadal
from src.data.split import split_and_scale_batadal
from src.models.deep_learning import TimeSeriesDataset, LSTMModel, CNN1DModel, train_model
from src.models.automata import ProbabilisticAutomata

def plot_confusion_matrix(y_true, y_pred, model_name, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Normal", "Anomaly"], 
                yticklabels=["Normal", "Anomaly"])
    plt.title(f"{model_name} - Confusion Matrix")
    plt.ylabel("Gercek Deger")
    plt.xlabel("Tahmin Edilen")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"cm_{model_name.lower()}.png"))
    plt.close()

def plot_roc_curve(y_true, y_probs, model_name, save_dir):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Area = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'{model_name} - ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"roc_{model_name.lower()}.png"))
    plt.close()

def plot_automata_heatmap(automata, save_dir):
    vocab = sorted(list(automata.vocabulary))
    n = len(vocab)
    matrix = np.zeros((n, n))
    
    for i, s1 in enumerate(vocab):
        for j, s2 in enumerate(vocab):
            matrix[i, j] = automata.transitions.get(s1, {}).get(s2, 0.0)
            
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, xticklabels=vocab, yticklabels=vocab, cmap="YlOrRd")
    plt.title("Olasiliksal Otomata Durum Gecis Matrisi (Heatmap)")
    plt.ylabel("Guncel Durum (Current State)")
    plt.xlabel("Sonraki Durum (Next State)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "automata_heatmap.png"))
    plt.close()

def main():
    save_dir = "figures"
    os.makedirs(save_dir, exist_ok=True)
    
    config = load_config()
    batadal_df = load_batadal(config['data']['batadal_path'])
    b_train, b_val, b_test, b_feats, b_target = split_and_scale_batadal(batadal_df)
    
    window_size = config['params']['window_size']
    alphabet_size = config['params']['alphabet_size']
    batch_size = config['params']['batch_size']
    
    train_dataset = TimeSeriesDataset(b_train[b_feats].values, b_train[b_target].values, window_size)
    test_dataset = TimeSeriesDataset(b_test[b_feats].values, b_test[b_target].values, window_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = len(b_feats)
    
    print("\nGrafikler cizdiriliyor (Hizli demo amacli 1 Seed ve 3 Epoch kullaniliyor)...")
    
    # 1. Derin Ogrenme Modelleri Grafikleri
    for model_name, ModelClass in [("LSTM", LSTMModel), ("1D-CNN", CNN1DModel)]:
        print(f"{model_name} modeli egitiliyor ve grafikleri ciziliyor...")
        if model_name == "LSTM":
            model = ModelClass(input_size=input_size)
        else:
            model = ModelClass(input_size=input_size, window_size=window_size)
            
        # Grafikleri cizmek icin hizli bir egitim
        trained_model = train_model(model, train_loader, train_loader, epochs=3, patience=3, device=device)
        
        trained_model.eval()
        all_probs, all_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = trained_model(X_batch.to(device)).cpu().numpy()
                all_probs.extend(outputs)
                all_targets.extend(y_batch.numpy())
                
        all_preds = (np.array(all_probs) > 0.5).astype(int)
        
        plot_confusion_matrix(all_targets, all_preds, model_name, save_dir)
        plot_roc_curve(all_targets, all_probs, model_name, save_dir)
        print(f"-> [{model_name}] Confusion Matrix ve ROC '{save_dir}/' klasorune eklendi.")

    # 2. Otomata Grafikleri
    print("\nOtomata egitiliyor ve Heatmap ciziliyor...")
    automata = ProbabilisticAutomata(window_size=window_size, alphabet_size=alphabet_size)
    automata.fit(b_train['PC1'].values)
    plot_automata_heatmap(automata, save_dir)
    print(f"-> [Otomata] Durum Gecis Matrisi (Heatmap) '{save_dir}/' klasorune eklendi.")
    
    print(f"\n[BASARILI] Tum grafikler '{save_dir}' klasorune basariyla kaydedildi!")

if __name__ == "__main__":
    main()
