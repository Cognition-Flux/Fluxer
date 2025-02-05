# %%
import os
import sys

# os.chdir("/home/alejandro/Desktop/repos/aiwtr")
# sys.path.append("/home/alejandro/Desktop/repos/aiwtr/src/forecasting.py")
import optuna
from optuna.samplers import TPESampler
from sklearn.preprocessing import StandardScaler
from optuna_dashboard import run_server


from src.forecasting import (
    generate_ts,
    objective,
    device,
    print_elapsed_time,
    sliding_window,
    create_dataloaders,
    TransformerForecaster,
    visualize_predictions,
)


# from src.timeseries_load import df_series as series
import time
import numpy as np
from pyentrp import entropy as ent
import warnings

warnings.filterwarnings("ignore")
series = []

# series[0]

sim = generate_ts(7 * 24 * 60, add_trend=True, add_noise=True).reset_index(
    drop=False, names=["time"]
)

series.append(sim)

# series_index = 2  # -1

for series_index in range(len(series)):
    storage = (
        # optuna.storages.InMemoryStorage()
        "sqlite:///optuna_dbs/Transformer_hyperparams_opt.sqlite3"
    )

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
    # run_server(storage)
    study.optimize(
        lambda trial: objective(trial, series_index=series_index, series=series),
        n_trials=50,
        n_jobs=1,
    )
    # run_server(storage, host="localhost", port=8080)
    # Start Optuna Dashboard

    print(
        "----------------------------------------------Best hyperparameters: ",
        study.best_params,
    )
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
        save_path=f"torch_models/best_model_{series[series_index].columns[1]}",
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
        save_path="figuras/" + series[series_index].columns[1] + ".png",
        num_samples=5,
        full_length_ts=ts,
    )
