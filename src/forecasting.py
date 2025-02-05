# %%
import os
import numpy as np
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# from ainwater.forecasts.POCs.read_data import df_series as series
# from ainwater.forecasts.POCs.read_data import ts_entropy
import time
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from datetime import datetime, timedelta
import warnings
import math
from pyentrp import entropy as ent

torch.cuda.init()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device=}")


def visualize_predictions(
    path_best_model,
    test_loader,
    ts_target_len,
    num_samples=5,
    main_title=None,
    MSE: float = None,
    full_length_ts: list = None,
    entropy: str = None,
    save_path: str = None,
):
    model = torch.load(path_best_model)
    model.eval()
    x_test, y_test = next(iter(test_loader))
    x_test, y_test = x_test.to(device), y_test.to(device)

    with torch.no_grad():
        predicted = model.predict(x_test, ts_target_len)

    fig, ax = plt.subplots(
        nrows=num_samples + 1, ncols=1, figsize=(11, 4 * (num_samples + 1) / 2)
    )
    if not full_length_ts:
        # Generate random time series for the first row
        full_length_ts_length = random.randint(
            50, 200
        )  # Arbitrary length between 50 and 200
        full_length_ts = np.random.randn(full_length_ts_length).tolist()

    # Plot random time series in the first row
    ax[0].plot(
        range(len(full_length_ts)),
        full_length_ts,
        color="purple",
        linewidth=1,
        label="Train/Test Dataset ",
    )
    # ax[0].set_title("Full-length Time Series", loc="left", fontweight="bold")
    ax[0].legend()

    # Generate a list of unique random indices
    total_samples = x_test.size(0)
    unique_indices = random.sample(
        range(total_samples), min(num_samples, total_samples)
    )

    for i, col in enumerate(ax[1:], start=1):
        r = unique_indices[i - 1]
        in_seq = x_test[r, :, 0].cpu().numpy()
        target_seq = y_test[r, :, 0].cpu().numpy()
        pred_seq = predicted[r, :, 0].cpu().numpy()
        x_axis = range(len(in_seq) + len(target_seq))
        col.set_title(f"Test Sample: {r}", loc="left")
        col.plot(x_axis[: len(in_seq)], in_seq, color="blue", label="Input")
        col.plot(x_axis[len(in_seq) :], target_seq, color="green", label="Target")
        col.plot(
            x_axis[len(in_seq) :],
            pred_seq,
            color="red",
            linestyle="--",
            label="Predicted",
        )
        col.axvline(x=len(in_seq), color="k", linestyle="--")
        col.legend()

    if main_title:
        fig.suptitle(
            f"{main_title} - Average test loss: {MSE:.4f} - Entropy: {entropy}",
            fontsize=11,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")


def print_elapsed_time(start, description):
    elapsed_time = time.time() - start
    print(f"{description}: {elapsed_time:.2f} seconds")


def generate_ts(
    len,
    freq_cycles_factor=20,
    amplitude_variation=0.5,
    add_trend=False,
    trend_strength=320,
    trend_cycle_length=20,
    add_noise=False,
):
    # Increase the frequency of oscillations
    t = np.linspace(0.0, 80 * np.pi, len)

    # Create a variable amplitude factor
    amplitude = 1 + amplitude_variation * np.sin(2 * t)

    # Create cyclical trend direction
    trend_direction = np.sin(2 * np.pi * t / trend_cycle_length)

    # Create global trend

    # Multiply t by compression_factor in the sine and cosine terms
    y = amplitude * (
        np.sin(5 * freq_cycles_factor * t) ** 2
        + 1 * np.cos(1 * freq_cycles_factor * t) ** 3
        + (np.sin(5 * freq_cycles_factor * t) * np.cos(1 * freq_cycles_factor * t))
        # + 0* np.sin(1 * freq_cycles_factor * t) ** 5  # New cyclical process
        # + np.random.normal(0, 0, len)
        + 2
    )
    if add_noise:
        y += np.random.normal(0, 1 * np.std(y) / 3, len)

    # Add global trend to the time series
    if add_trend:
        global_trend = trend_strength * np.cumsum(trend_direction) / len
        y += global_trend

    # Generate timestamp index
    start_time = datetime.now().replace(microsecond=0)
    time_index = [start_time + timedelta(minutes=i) for i in range(len)]

    # Create pandas DataFrame with timestamp index
    return pd.DataFrame(y, index=time_index, columns=["simulation"])


def rewrite_timestamps(df):
    # Ensure the dataframe is not empty
    if df.empty or df.columns.empty:
        return df

    # Get the first column
    first_column = df.columns[0]

    # Read the first entry and parse it as a datetime
    start_time = pd.to_datetime(df.iloc[0, 0])

    # Generate new timestamps
    new_timestamps = [
        (start_time + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S")
        for i in range(len(df))
    ]

    # Update the column with new timestamps as strings
    df[first_column] = new_timestamps

    # Ensure the dtype is 'object'
    # df[first_column] = df[first_column].astype("object")

    return df


class TimeSeriesDataset(Dataset):
    def __init__(self, X, Y):
        self.X = (
            torch.tensor(X).float().unsqueeze(2)
        )  # Add an extra dimension for features
        self.Y = (
            torch.tensor(Y).float().unsqueeze(2)
        )  # Add an extra dimension for features

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def create_dataloaders(X, Y, batch_size, test_ds_len, device):
    ds_len = len(X)
    train_dataset = TimeSeriesDataset(
        X[: ds_len - test_ds_len], Y[: ds_len - test_ds_len]
    )
    test_dataset = TimeSeriesDataset(
        X[ds_len - test_ds_len :], Y[ds_len - test_ds_len :]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def sliding_window(ts, features, target_len=1):
    X = []
    Y = []

    for i in range(features + target_len, len(ts) + 1):
        X.append(ts[i - (features + target_len) : i - target_len])
        Y.append(ts[i - target_len : i])

    return X, Y


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0), :]


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_size,
        d_model,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        dropout=0.1,
    ):
        super(TransformerModel, self).__init__()
        self.input_size = input_size
        self.d_model = d_model

        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.output_proj = nn.Linear(d_model, input_size)

    def forward(self, src, tgt):
        src = self.input_proj(src)
        src = self.pos_encoder(src)

        tgt = self.input_proj(tgt)
        tgt = self.pos_encoder(tgt)

        output = self.transformer(src, tgt)
        return self.output_proj(output)


class TransformerForecaster(nn.Module):
    def __init__(
        self,
        hidden_size,
        input_size=1,
        output_size=1,
        num_layers=1,
        dropout=0.15,
        weight_decay=0.05,
    ):
        super(TransformerForecaster, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.transformer = TransformerModel(
            input_size=input_size,
            d_model=hidden_size,
            nhead=8,  # Number of attention heads
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
        )

        self.weight_decay = weight_decay

    def train_model(
        self,
        train_loader,
        test_loader,
        epochs,
        lr=0.01,
        ts_target_len=1,
        save_path=None,
        trial=None,
        last_best_model_path=None,
    ):
        train_losses = torch.full((epochs,), float("nan"))
        test_losses = torch.full((epochs,), float("nan"))
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=lr, weight_decay=self.weight_decay
        )
        criterion = nn.MSELoss()
        best_test_loss = float("inf")

        for e in range(epochs):
            self.train()
            epoch_loss = 0
            for batch_idx, (train, target) in enumerate(train_loader):
                train, target = train.to(device), target.to(device)
                optimizer.zero_grad()

                # Prepare input for transformer
                src = train.transpose(0, 1)
                tgt = torch.zeros_like(target).transpose(0, 1)

                predicted = self.transformer(src, tgt)
                predicted = predicted.transpose(0, 1)

                loss = criterion(predicted, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)
            train_losses[e] = avg_train_loss

            avg_test_loss = self.evaluate(test_loader, ts_target_len)
            test_losses[e] = avg_test_loss

            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                if save_path:
                    last_best_model_path = f"{save_path}_best_test_loss"
                    torch.save(self, last_best_model_path)
                print(
                    f"---NEW best model---Model saved at epoch {e} with test loss: {best_test_loss:.4f}"
                )

            if e % 2 == 0:
                print(
                    f"Epoch {e}/{epochs}: Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}"
                )

            if trial is not None:
                trial.report(avg_test_loss, e)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        return train_losses, test_losses, last_best_model_path, best_test_loss

    def predict(self, x, target_len):
        self.eval()
        x = x.to(device)
        batch_size, seq_len, _ = x.size()

        src = x.transpose(0, 1)
        tgt = torch.zeros(target_len, batch_size, self.input_size, device=device)

        with torch.no_grad():
            output = self.transformer(src, tgt)

        return output.transpose(0, 1)

    def evaluate(self, test_loader, ts_target_len):
        self.eval()
        test_loss = 0
        criterion = nn.MSELoss()
        with torch.no_grad():
            for batch_idx, (x_test, y_test) in enumerate(test_loader):
                x_test, y_test = x_test.to(device), y_test.to(device)
                y_pred = self.predict(x_test, ts_target_len)
                loss = criterion(y_pred, y_test)
                test_loss += loss.item()
        avg_test_loss = test_loss / len(test_loader)
        return avg_test_loss


# The rest of your code (TimeSeriesDataset, create_dataloaders, sliding_window, objective) remains the same


def objective(trial, series_index, series):
    largo_dataset = trial.suggest_int("largo_dataset", 8, 2 * 24, step=8)
    hidden_size = trial.suggest_int("hidden_size", 8, 8 * 8, step=8)
    num_layers = trial.suggest_int("num_layers", 2, 4)
    ts_history_len = trial.suggest_int("ts_history_len", 30, 1 * 60, step=30)
    ts_target_len = trial.suggest_int("ts_target_len", 15, 30, step=15)
    batch_size = trial.suggest_int("batch_size", 16, 4 * 8, step=8)
    learning_rate = trial.suggest_float("learning_rate", 0.001, 0.01, step=0.01)
    test_set_fraction = 0.2  #
    data_prep_start = time.time()
    ts = ((series[series_index]).iloc[:, 1]).tolist()
    ts = ts[-largo_dataset * 60 :]
    data = np.array(ts)
    data = data.reshape(-1, 1)
    standard_scaler = StandardScaler()
    ts = standard_scaler.fit_transform(data).reshape(-1).tolist()
    print_elapsed_time(data_prep_start, "Data preparation")

    test_ds_len = int(len(ts) * test_set_fraction)
    epochs = 5
    # Prepare the data
    data_process_start = time.time()
    X, Y = sliding_window(ts, ts_history_len, ts_target_len)
    # batch_size = 2 * 8
    train_loader, test_loader = create_dataloaders(
        X, Y, batch_size, test_ds_len, device
    )
    print_elapsed_time(data_process_start, "Data processing")

    # Initialize the model
    model_init_start = time.time()
    model = TransformerForecaster(
        hidden_size=hidden_size, input_size=1, output_size=1, num_layers=num_layers
    ).to(device)
    print_elapsed_time(model_init_start, "Model initialization")

    # Train the model
    training_start = time.time()
    model.train()
    train_losses, test_losses, best_model_path, best_test_loss = model.train_model(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=epochs,
        lr=learning_rate,
        save_path=None,  # f"forecast/trained_torch_models/best_model_{ts_name}.pth",
        trial=trial,
    )

    print_elapsed_time(training_start, "Model training")
    return test_losses[-1]


if __name__ == "__main__":
    series = []
    sim = generate_ts(7 * 24 * 60, add_trend=True, add_noise=True).reset_index(
        drop=False, names=["time"]
    )

    series.append(sim)

    series_index = -1
    storage = (
        optuna.storages.InMemoryStorage()
    )  # "sqlite:///forecast/optuna_dbs/Transformer_hyperparams_opt.sqlite3"

    try:
        optuna.delete_study(study_name=series[series_index].columns[1], storage=storage)
    except Exception:
        print(f"estudio '{series[series_index].columns[1]}' no estaba guardado")

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
        storage=storage,
        study_name=series[series_index].columns[1],
        load_if_exists=True,
        sampler=TPESampler(),
    )
    study.optimize(
        lambda trial: objective(trial, series_index=series_index, series=series),
        n_trials=50,
        n_jobs=1,
    )

    print(
        "----------------------------------------------Best hyperparameters: ",
        study.best_params,
    )

    ############################################
    # best_params = optuna.load_study(
    #     study_name=series[series_index].columns[1], storage=storage
    # ).best_params

    # best_trial_MSE = optuna.load_study(
    #     study_name=series[series_index].columns[1], storage=storage
    # ).best_trial.values[0]
    best_params = study.best_params
    best_trial_MSE = study.best_trial.values[0]

    data_prep_start = time.time()
    ts = ((series[series_index]).iloc[:, 1]).tolist()  #

    ts = ts[-best_params["largo_dataset"] * 60 :]
    data = np.array(ts)
    data = data.reshape(-1, 1)
    standard_scaler = StandardScaler()
    ts = standard_scaler.fit_transform(data).reshape(-1).tolist()
    # print(f"largo de la serie {len(ts)=}")
    print_elapsed_time(data_prep_start, "Data preparation")

    # Model parameters
    # hidden_size = int(1 * 8 / 1)
    test_ds_len = int(len(ts) * 0.2)  # best_params["test_set_fraction"])
    epochs = 200
    # ts_history_len = 1 * 60
    # ts_target_len = 1 * 30

    # Prepare the data
    data_process_start = time.time()
    ts_target_len = best_params["ts_target_len"]
    X, Y = sliding_window(ts, best_params["ts_history_len"], ts_target_len)
    # batch_size = 2 * 8
    train_loader, test_loader = create_dataloaders(
        X, Y, best_params["batch_size"], test_ds_len, device
    )
    print_elapsed_time(data_process_start, "Data processing")

    # Initialize the model
    model_init_start = time.time()
    print(f"{device=}")
    model = TransformerForecaster(
        hidden_size=best_params["hidden_size"],
        input_size=1,
        output_size=1,
        num_layers=best_params["num_layers"],
    ).to(device)
    print_elapsed_time(model_init_start, "Model initialization")

    # Train the model
    training_start = time.time()
    model.train()
    train_losses, test_losses, best_model_path, best_test_loss = model.train_model(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=epochs,
        # method="teacher_forcing",  # "mixed_teacher_forcing",
        # tfr=0.01,
        lr=best_params["learning_rate"],
        # dynamic_tf=False,
        ts_target_len=ts_target_len,  # best_params["ts_target_len"],
        save_path=f"forecast/torch_models/best_model_{series[series_index].columns[1]}",
        trial=None,
    )

    print_elapsed_time(training_start, "Model training")
    # # %%
    # # best_model_path = "forecast/trained_torch_models/best_model_rssi.24e124454d021682.pth_best_test_loss_0.33"
    # best_model_path = (
    #     "forecast/trained_torch_models/best_model_simulation.pth_best_test_loss"
    # )

    visualize_predictions(
        best_model_path,
        test_loader,
        ts_target_len,  # best_params["ts_target_len"],
        main_title=series[series_index].columns[1],
        MSE=test_losses[-1],  # best_test_loss,
        entropy=(
            round(ent.shannon_entropy(ts), 3),
            round(ent.permutation_entropy(ts), 3),
        ),
        save_path="forecast/figures/" + series[series_index].columns[1] + ".png",
        num_samples=5,
        full_length_ts=ts,
    )
