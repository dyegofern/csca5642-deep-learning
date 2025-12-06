# Hyperparameter Tuning Configuration
RUN_HYPERPARAMETER_TUNING = True  # Set to True to run Optuna optimization
N_TUNING_TRIALS = 20  # Number of Optuna trials (more = better but slower)
TUNING_TIMEOUT = 300  # Maximum time for tuning in seconds (60 minutes)

# Path to save/load best hyperparameters
HYPERPARAMS_PATH = os.path.join(MODEL_DIR, 'best_hyperparameters.json')

print(f"Hyperparameter tuning: {'ENABLED' if RUN_HYPERPARAMETER_TUNING else 'DISABLED'}")
print(f"Trials: {N_TUNING_TRIALS}, Timeout: {TUNING_TIMEOUT}s")
print(f"Hyperparameters path: {HYPERPARAMS_PATH}")