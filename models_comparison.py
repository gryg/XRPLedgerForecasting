import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import warnings
import os
from datetime import datetime
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class OutputManager:
    def __init__(self, base_dir="outputs"):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(base_dir, f"run_{self.timestamp}")
        self.create_directories()
        
    def create_directories(self):
        """Create output directory structure"""
        directories = [
            "metrics",
            "plots",
            "model",
            "predictions"
        ]
        
        for dir_name in directories:
            os.makedirs(os.path.join(self.run_dir, dir_name), exist_ok=True)
    
    def save_metrics(self, metrics_dict, dataset_type):
        """Save metrics to JSON file"""
        file_path = os.path.join(self.run_dir, "metrics", f"{dataset_type}_metrics.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_dict, f, indent=4, ensure_ascii=False)
    
    def save_plot(self, fig, plot_name):
        """Save matplotlib figure"""
        file_path = os.path.join(self.run_dir, "plots", f"{plot_name}.png")
        fig.savefig(file_path)
        plt.close(fig)
    
    def save_predictions(self, predictions, dataset_type):
        """Save predictions to CSV"""
        file_path = os.path.join(self.run_dir, "predictions", f"{dataset_type}_predictions.csv")
        pd.DataFrame(predictions).to_csv(file_path, encoding='utf-8')
    
    def save_model_summary(self, model, model_name):
        """Save model architecture summary with proper encoding"""
        try:
            file_path = os.path.join(self.run_dir, "model", f"{model_name}_summary.txt")
            with open(file_path, 'w', encoding='utf-8') as f:
                # Only save summary for Keras models
                if hasattr(model, 'summary'):
                    # Capture the summary string
                    stringlist = []
                    model.summary(print_fn=lambda x: stringlist.append(x))
                    summary_string = '\n'.join(stringlist)
                    f.write(summary_string)
                else:
                    # For non-Keras models, save basic info
                    f.write(f"Model: {model_name}\n")
                    f.write(f"Type: {type(model).__name__}\n")
                    if hasattr(model, 'get_params'):
                        f.write("\nParameters:\n")
                        for param, value in model.get_params().items():
                            f.write(f"{param}: {value}\n")
        except Exception as e:
            logger.warning(f"Could not save model summary for {model_name}: {str(e)}")
            
class SimpleLSTM:
    def __init__(self, hidden_size=64):
        self.hidden_size = hidden_size
        self.model = None
        self.scaler = StandardScaler()
    
    def create_model(self, lookback, forecast_horizon, num_features):
        """Create simple LSTM model"""
        model = Sequential([
            LSTM(self.hidden_size, input_shape=(lookback, num_features)),
            BatchNormalization(),
            Dropout(0.2),
            Dense(forecast_horizon)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

class ImprovedDeepARModel:
    def __init__(self, hidden_size=40, num_layers=2, dropout_rate=0.2):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.model = None
        self.scaler = StandardScaler()
    
    def create_model(self, lookback, forecast_horizon, num_features):
        """Create DeepAR model with proper loss handling"""
        inputs = Input(shape=(lookback, num_features))
        x = BatchNormalization()(inputs)
        
        for i in range(self.num_layers):
            x = Bidirectional(
                LSTM(
                    self.hidden_size,
                    return_sequences=(i < self.num_layers - 1),
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.dropout_rate/2
                )
            )(x)
            x = BatchNormalization()(x)
            
            if i < self.num_layers - 1:
                x = Dropout(self.dropout_rate)(x)
        
        # Single output layer for predictions
        predictions = Dense(forecast_horizon)(x)
        
        self.model = Model(inputs=inputs, outputs=predictions)
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return self.model
    
class CombinedForecaster:
    def __init__(self, lookback=30, horizon=7):
        self.lookback = lookback
        self.horizon = horizon
        self.scaler = MinMaxScaler()
        self.models = {}
        self.histories = {}
        self.predictions = {}
        self.metrics = {}
        self.output_manager = OutputManager()
        self.deepar = ImprovedDeepARModel()
        self.simple_lstm = SimpleLSTM()
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive set of metrics"""
        # Ensure arrays are properly shaped
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)
        
        metrics = {
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred)
        }
        
        # Calculate MAPE only for non-zero true values
        mask = y_true != 0
        if np.any(mask):
            metrics['MAPE'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            metrics['MAPE'] = np.nan
            
        # Calculate sMAPE
        denominator = np.abs(y_true) + np.abs(y_pred)
        mask = denominator != 0
        if np.any(mask):
            metrics['sMAPE'] = np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]) * 100
        else:
            metrics['sMAPE'] = np.nan
        
        return metrics
    
    def prepare_data(self, df, target_col='Closed_Ledgers'):
        """Prepare data for time series forecasting"""
        features = df[[target_col] + [col for col in df.columns if col.startswith(('Lag_', 'Day_MA'))]].copy()
        
        data = features.values
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        data = np.clip(data, q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(len(scaled_data) - self.lookback - self.horizon + 1):
            X.append(scaled_data[i:(i + self.lookback)])
            y.append(scaled_data[i + self.lookback:i + self.lookback + self.horizon, 0])
            
        return np.array(X), np.array(y)
    
    def calculate_prediction_intervals(self, predictions, confidence=0.8, model_name=None):
        """Calculate prediction intervals using residual bootstrap"""
        if isinstance(predictions, list):
            predictions = np.array(predictions)
        
        # Handle single sequence vs full dataset
        is_single_sequence = len(predictions.shape) == 1
        if is_single_sequence:
            predictions = predictions.reshape(1, -1)
        
        # Calculate residuals by sequence
        residuals = []
        if model_name and model_name in self.predictions:
            full_predictions = self.predictions[model_name]
            for i in range(len(self.y_val)):
                seq_residuals = self.y_val[i] - full_predictions[i]
                residuals.extend(seq_residuals)
        else:
            # Fallback to using provided predictions directly
            seq_residuals = self.y_val.reshape(-1) - predictions.reshape(-1)
            residuals = seq_residuals
        
        residuals = np.array(residuals)
        
        # Bootstrap prediction intervals
        n_bootstrap = 1000
        bootstrapped_predictions = []
        
        for _ in range(n_bootstrap):
            # Sample residuals and reshape to match predictions
            sampled_residuals = np.random.choice(residuals, size=predictions.shape)
            bootstrapped_predictions.append(predictions + sampled_residuals)
        
        bootstrapped_predictions = np.array(bootstrapped_predictions)
        
        # Calculate intervals
        lower_bound = np.percentile(bootstrapped_predictions, (1 - confidence) * 50, axis=0)
        upper_bound = np.percentile(bootstrapped_predictions, (1 + confidence) * 50, axis=0)
        
        if is_single_sequence:
            lower_bound = lower_bound.flatten()
            upper_bound = upper_bound.flatten()
        
        return lower_bound, upper_bound

    def train_models(self, df, val_split=0.2):
        """Train all models"""
        # Prepare data
        X, y = self.prepare_data(df)
        split_idx = int(len(X) * (1 - val_split))
        
        self.X_train, self.X_val = X[:split_idx], X[split_idx:]
        self.y_train, self.y_val = y[:split_idx], y[split_idx:]
        
        # Train DeepAR
        print("Training DeepAR...")
        deepar_model = self.deepar.create_model(self.lookback, self.horizon, X.shape[2])
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.0001
        )
        
        history_deepar = deepar_model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=50,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        self.models['DeepAR'] = deepar_model
        self.histories['DeepAR'] = history_deepar.history

        # Train Simple LSTM
        print("Training Simple LSTM...")
        lstm_model = self.simple_lstm.create_model(self.lookback, self.horizon, X.shape[2])
        history_lstm = lstm_model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=50,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        self.models['SimpleLSTM'] = lstm_model
        self.histories['SimpleLSTM'] = history_lstm.history
        
        # Train other models
        for model_name, model_config in [
            ('XGBoost', XGBRegressor(n_estimators=100, learning_rate=0.1)),
            ('RandomForest', RandomForestRegressor(n_estimators=100))
        ]:
            print(f"Training {model_name}...")
            X_train_2d = self.X_train.reshape(self.X_train.shape[0], -1)
            X_val_2d = self.X_val.reshape(self.X_val.shape[0], -1)
            
            model_config.fit(X_train_2d, self.y_train)
            self.models[model_name] = model_config
            
            # Generate predictions
            if model_name in ['XGBoost', 'RandomForest']:
                self.predictions[model_name] = model_config.predict(X_val_2d)
        
        # Generate neural network predictions
        self.predictions['DeepAR'] = deepar_model.predict(self.X_val)
        self.predictions['SimpleLSTM'] = lstm_model.predict(self.X_val)
        
        # Calculate prediction intervals and metrics for all models
        for model_name in self.predictions:
            pred = self.predictions[model_name]
            self.metrics[model_name] = self.calculate_metrics(self.y_val, pred)
        
        # Save model summaries
        for model_name, model in self.models.items():
            if hasattr(model, 'summary'):
                self.output_manager.save_model_summary(model, model_name)
    
    def plot_metrics(self):
        """Create and save detailed metrics plots"""
        # Define distinct colors for each model
        model_colors = {
            'XGBoost': '#FF6B6B',      # Coral red
            'RandomForest': '#4ECDC4',  # Turquoise
            'DeepAR': '#45B7D1',       # Sky blue
            'SimpleLSTM': '#96CEB4'     # Sage green
        }
        
        # Create figure for metrics plots
        fig = plt.figure(figsize=(20, 12))
        
        # List of metrics to plot
        metrics_to_plot = {
            'RMSE': 'RMSE Comparison',
            'MAE': 'MAE Comparison',
            'MAPE': 'MAPE Comparison',
            'sMAPE': 'sMAPE Comparison',
            'R2': 'R² Comparison'
        }
        
        # Plot each metric
        for idx, (metric, title) in enumerate(metrics_to_plot.items(), 1):
            plt.subplot(2, 3, idx)
            
            values = [metrics[metric] for metrics in self.metrics.values()]
            models = list(self.metrics.keys())
            
            bars = plt.bar(models, 
                        values, 
                        color=[model_colors[model] for model in models])
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom')
            
            plt.title(title)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            # Add appropriate y-label
            if metric in ['MAPE', 'sMAPE']:
                plt.ylabel('Percentage (%)')
            elif metric == 'R2':
                plt.ylabel('R² Score')
            else:
                plt.ylabel('Error')
        
        # Add metrics summary table
        plt.subplot(2, 3, 6)
        plt.axis('off')
        
        # Prepare table data
        metrics_list = list(metrics_to_plot.keys())
        table_data = [['Model'] + metrics_list]
        
        for model in self.metrics:
            row = [model] + [f'{self.metrics[model][metric]:.4f}' for metric in metrics_list]
            table_data.append(row)
        
        # Create and style table
        table = plt.table(cellText=table_data,
                        loc='center',
                        cellLoc='center',
                        colWidths=[0.2] + [0.15] * len(metrics_list))
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Style headers
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#E6E6E6')
        
        plt.title('Metrics Summary Table', pad=20)
        
        plt.tight_layout()
        
        # Save the metrics plot
        self.output_manager.save_plot(fig, 'detailed_metrics')
    
    def plot_results(self, num_samples=7):
        """Create and save comprehensive visualization plots with improved colors and legend"""
        # Define distinct colors for each model
        model_colors = {
            'XGBoost': '#FF6B6B',      # Coral red
            'RandomForest': '#4ECDC4',  # Turquoise
            'DeepAR': '#45B7D1',       # Sky blue
            'SimpleLSTM': '#96CEB4'     # Sage green
        }
        
        # Create multiple sequence predictions plot
        fig_sequences = plt.figure(figsize=(20, 4*num_samples))
        
        for i in range(num_samples):
            plt.subplot(num_samples, 1, i+1)
            plt.plot(self.y_val[i], 'k-', label='Actual', linewidth=2)
            
            for model_name, pred in self.predictions.items():
                color = model_colors[model_name]
                pred_seq = pred[i] if len(pred.shape) > 1 else pred
                
                # Plot prediction line
                plt.plot(pred_seq, '--', 
                        label=f'{model_name} Prediction', 
                        color=color,
                        linewidth=1.5)
                
                try:
                    # Calculate and plot confidence intervals
                    lower_bound, upper_bound = self.calculate_prediction_intervals(
                        pred_seq, 
                        model_name=model_name
                    )
                    plt.fill_between(range(len(lower_bound)), 
                                lower_bound,
                                upper_bound,
                                color=color,
                                alpha=0.2,
                                label=f'{model_name} (80% Confidence Interval)')
                except Exception as e:
                    logger.warning(f"Could not calculate intervals for {model_name}: {str(e)}")
            
            plt.title(f'Prediction Sequence {i+1}')
            plt.xlabel('Time Steps')
            plt.ylabel('Normalized Volume')
            
            # Only show legend for the first subplot
            if i == 0:

                legend = plt.legend(bbox_to_anchor=(1.05, 1), 
                                loc='upper left',
                                borderaxespad=0.,
                                frameon=True,
                                fancybox=True,
                                shadow=True)
                plt.setp(legend.get_texts(), fontsize='10')
            
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.output_manager.save_plot(fig_sequences, 'prediction_sequences')
        
        # Plot metrics comparison with the same color scheme
        fig_metrics = plt.figure(figsize=(20, 10))
        
        # Metrics comparison
        plt.subplot(221)
        metrics_df = pd.DataFrame(self.metrics).T[['RMSE', 'MAE', 'MAPE']]
        ax = metrics_df.plot(kind='bar', 
                            color=[model_colors[model] for model in metrics_df.index])
        plt.title('Model Performance Metrics')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on the bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3)
        
        # Error distribution
        plt.subplot(222)
        for model_name, pred in self.predictions.items():
            errors = self.y_val.reshape(-1) - pred.reshape(-1)
            sns.kdeplot(errors, 
                    label=model_name, 
                    color=model_colors[model_name])
        plt.title('Error Distribution')
        plt.xlabel('Error')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # R² comparison
        plt.subplot(223)
        r2_scores = {model: metrics['R2'] for model, metrics in self.metrics.items()}
        bars = plt.bar(r2_scores.keys(), 
                    r2_scores.values(),
                    color=[model_colors[model] for model in r2_scores.keys()])
        plt.title('R² Score Comparison')
        plt.ylabel('R² Score')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Add value labels on the bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')
        
        # Actual vs Predicted scatter plot
        plt.subplot(224)
        for model_name, pred in self.predictions.items():
            plt.scatter(self.y_val.flatten(), 
                    pred.flatten(),
                    alpha=0.5,
                    label=model_name,
                    color=model_colors[model_name])
        plt.plot([self.y_val.min(), self.y_val.max()], 
                [self.y_val.min(), self.y_val.max()],
                'k--', label='Perfect Prediction')
        plt.title('Actual vs Predicted Values')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.output_manager.save_plot(fig_metrics, 'model_metrics')
        self.plot_metrics()
            
        
def main():
    try:
        # Load data
        df = pd.read_csv('normalized_transaction_data.csv')
        
        # Initialize and train forecaster
        print("Initializing forecaster...")
        forecaster = CombinedForecaster(lookback=30, horizon=7)
        
        # Train models
        print("Training models...")
        forecaster.train_models(df)
        
        # Plot and save results
        print("Generating and saving plots...")
        forecaster.plot_results()
        
        print(f"Results saved in: {forecaster.output_manager.run_dir}")
        return forecaster
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    forecaster = main()