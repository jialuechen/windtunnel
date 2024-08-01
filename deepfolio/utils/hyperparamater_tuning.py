import tensorflow as tf
from kerastuner import RandomSearch
from models.diffopt_portfolio import DiffOptPortfolio
def build_model(hp):
    model = DiffOptPortfolio(
        input_dim=hp.Int('input_dim', 10, 100),
        n_assets=hp.Int('n_assets', 5, 50),
        hidden_dim=hp.Int('hidden_dim', 32, 256, step=32)
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
        loss='mse'
    )
    return model

def tune_hyperparameters(x, y, epochs=10, max_trials=10):
    tuner = RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=max_trials,
        executions_per_trial=3,
        directory='hyperparam_tuning',
        project_name='diffopt_portfolio'
    )
    
    tuner.search(x, y, epochs=epochs, validation_split=0.2)
    
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    return best_model, best_hyperparameters