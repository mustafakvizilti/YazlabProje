import torch
from torch.utils.data import DataLoader
from src.data.preprocess import load_config, load_batadal, load_and_concat_skab
from src.data.split import split_and_scale_batadal, get_skab_splits
from src.models.deep_learning import TimeSeriesDataset, LSTMModel, train_model
from src.models.automata import ProbabilisticAutomata
import json

def main():
    config = load_config()

    batadal_df = load_batadal(config['data']['batadal_path'])
    skab_df = load_and_concat_skab(config['data']['skab_dir'])

    b_train, b_val, b_test, b_feats, b_target = split_and_scale_batadal(batadal_df)

    window_size = config['params']['window_size']
    alphabet_size = config['params']['alphabet_size']
    batch_size = config['params']['batch_size']
    max_epochs = config['params']['max_epochs']
    patience = config['params']['early_stopping']

    train_dataset = TimeSeriesDataset(b_train[b_feats].values, b_train[b_target].values, window_size)
    val_dataset = TimeSeriesDataset(b_val[b_feats].values, b_val[b_target].values, window_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_size = len(b_feats)
    lstm_model = LSTMModel(input_size=input_size)

    print("\nLSTM Modeli egitimi basliyor...")
    trained_lstm = train_model(
        model=lstm_model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=max_epochs,
        patience=patience,
        device=device
    )
    print("LSTM egitimi tamamlandi.")

    print("\n--- OLASILIKSAL OTOMATA TESTI ---")
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