import xgboost as xgb
print(f"XGBoost version: {xgb.__version__}")

# Test GPU support
params = {'tree_method': 'gpu_hist'}  # GPU-specific tree method
dtrain = xgb.DMatrix([[1, 2], [3, 4]], label=[1, 0])
booster = xgb.train(params, dtrain, num_boost_round=1)

print("GPU is enabled and working!")

