import os
import re
import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ================= USER CONFIG =================

DATA_PATH = r"D:\ML Repositories\marketdata\data\raw\processed\Azadpur\Delicious_A_dataset.csv"
EXPERIMENT_DIR = r"D:\ML Repositories\marketdata\experiments\2025-08-20_15-05-55_Azadpur_Delicious_A_all_precut_60d"

SEASON_START = "09-01"
SEASON_END   = "03-31"
TEST_YEAR    = 2025
MAX_LAG = 40

MODEL_DIR = os.path.join(EXPERIMENT_DIR,"models")


# ================= HELPERS =================

def season_bounds(year,start,end):

    s = pd.Timestamp(f"{year}-{start}")

    if pd.Timestamp(f"2000-{end}") > pd.Timestamp(f"2000-{start}"):
        e = pd.Timestamp(f"{year}-{end}") + pd.Timedelta(days=1)
    else:
        e = pd.Timestamp(f"{year+1}-{end}") + pd.Timedelta(days=1)

    return s,e


def metrics(y_true,y_pred):

    mse = mean_squared_error(y_true,y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true,y_pred)
    mape = np.mean(np.abs((y_true-y_pred)/y_true))*100
    r2 = r2_score(y_true,y_pred)

    return rmse,mae,mape,r2


def build_scaled_lag_frame(df,scaler,max_lag):

    d = df.copy()

    d["y_scaled"] = scaler.transform(d[["y"]])
    d["y_scaled_filled"] = d["y_scaled"].ffill().fillna(0)

    for k in range(1,max_lag+1):
        d[f"y_lag{k}"] = d["y_scaled_filled"].shift(k)
        d[f"mask_lag{k}"] = d["mask"].shift(k)

    feats = [f"y_lag{i}" for i in range(1,max_lag+1)] + \
            [f"mask_lag{i}" for i in range(1,max_lag+1)]

    return d,feats


def recursive_tree_predict(model,df_test,max_lag):

    hist = df_test["y_scaled"].ffill().values.copy()
    preds = []

    for i in range(len(df_test)-max_lag+1):

        window = hist[i:i+max_lag]
        mask_window = df_test["mask"].iloc[i:i+max_lag].values

        x = np.concatenate([window[::-1],mask_window]).reshape(1,-1)

        # keep feature names if model expects
        if hasattr(model,"feature_names_in_"):
            x = pd.DataFrame(x,columns=model.feature_names_in_)

        yhat = model.predict(x)[0]

        preds.append(yhat)

        hist[i+max_lag-1] = yhat

    return np.array(preds)


# ===== REMOVE DUPLICATE MODEL RUNS =====

def select_latest_models(files):

    families = {}

    for f in files:

        base = re.split(r'_\d{4}-\d{2}-\d{2}_',f)[0]

        if base not in families:
            families[base] = f
        else:
            families[base] = max(families[base],f)

    return list(families.values())


# ================= LOAD DATA =================

print("\nLoading dataset")

df = pd.read_csv(DATA_PATH)

df = df.rename(columns={
    "Date":"ds",
    "Avg Price (per kg)":"y",
    "Mask":"mask"
})

df["ds"] = pd.to_datetime(df["ds"],format="mixed",dayfirst=True,errors="coerce")
df = df.dropna(subset=["ds"])
df = df.sort_values("ds").reset_index(drop=True)

print("Data range:",df["ds"].min(),"→",df["ds"].max())


TEST_START,TEST_END_EXCL = season_bounds(TEST_YEAR,SEASON_START,SEASON_END)
TEST_END = min(TEST_END_EXCL,df["ds"].max())

print("Test window:",TEST_START,"→",TEST_END)


train_obs = df[(df["ds"] < TEST_START) & (df["mask"]==1)]

scaler = StandardScaler()
scaler.fit(train_obs[["y"]])


buffer_start = TEST_START - pd.Timedelta(days=MAX_LAG+5)

eval_df = df[df["ds"]>=buffer_start].copy()
eval_df,FEATURES = build_scaled_lag_frame(eval_df,scaler,MAX_LAG)

test = eval_df[(eval_df["ds"]>=TEST_START)&(eval_df["ds"]<TEST_END)].copy()
test = test.dropna(subset=FEATURES)

print("Test rows:",len(test))


# ================= MODEL LIST =================

all_models = select_latest_models(os.listdir(MODEL_DIR))

print("\nModels selected:",len(all_models))


# ================= EVALUATION =================

results = []

for file in all_models:

    path = os.path.join(MODEL_DIR,file)

    print("\nEvaluating:",file)

    try:

        # TREE MODELS
        if file.endswith(".joblib"):

            model = joblib.load(path)
            pred_scaled = recursive_tree_predict(model,test,MAX_LAG)

        # LSTM
        elif file.endswith(".h5"):

            from tensorflow.keras.models import load_model

            model = load_model(path,compile=False)

            X = test[FEATURES].values.reshape(len(test),MAX_LAG,2)

            pred_scaled = model.predict(X,verbose=0).flatten()

        # DARTS
        elif ".darts.pt" in file:

            from darts.models import NBEATSModel,NHiTSModel
            from darts import TimeSeries

            torch.serialization.add_safe_globals([NBEATSModel,NHiTSModel])

            if "nbeats" in file:
                model = NBEATSModel.load(path,map_location=torch.device("cpu"))
            else:
                model = NHiTSModel.load(path,map_location=torch.device("cpu"))

            series = TimeSeries.from_dataframe(df,"ds","y")

            fc = model.predict(len(test),series)

            pred = fc.values().flatten()

            pred_scaled = scaler.transform(pred.reshape(-1,1)).flatten()

        # TORCH STATE_DICT MODELS
        elif file.endswith(".pt"):

            print("Torch state_dict model → skipping inference (architecture unknown)")
            continue

        # NeuralProphet / TBATS
        elif file.endswith(".bin"):

            print("External model bin → skipping (custom loader required)")
            continue

        else:
            continue


        pred = scaler.inverse_transform(pred_scaled.reshape(-1,1)).flatten()

        valid_len = len(pred)
        test_aligned = test.iloc[:valid_len]

        mask = test_aligned["mask"].values==1

        y_true = test_aligned.loc[mask,"y"].values
        y_pred = pred[mask]

        if len(y_true)==0:
            continue

        rmse,mae,mape,r2 = metrics(y_true,y_pred)

        results.append({
            "model":file,
            "RMSE":rmse,
            "MAE":mae,
            "MAPE":mape,
            "R2":r2
        })

    except Exception as e:

        print("FAILED:",file)
        print(e)


# ================= FINAL TABLE =================

if len(results)>0:

    res = pd.DataFrame(results).sort_values("RMSE")

    print("\nFINAL RANKING\n")
    print(res)

    res.to_csv(f"model_comparison_{TEST_YEAR}.csv",index=False)

else:
    print("\nNo models evaluated")