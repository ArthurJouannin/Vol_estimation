import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from arch import arch_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

class Vol:    
    def __init__(self, lookback=60, forecast_h=1):
        self.lookback = lookback
        self.forecast_h = forecast_h
        self.scaler = StandardScaler()
        self.har = None
        self.garch = None
        self.lstm = None
        self.ensemble = LinearRegression()
        self.indiv_predictions = {}
        self.predictions = None
        
    def load_data(self, csv_path):      
        data = pd.read_csv(csv_path)
        data['Date'] = pd.to_datetime(data['timestamp'])
        data.set_index('Date', inplace=True)
        self.prices = data['close'].dropna()
        self.rv()
        return self.prices
    
    def rv(self):
        returns = np.log(self.prices / self.prices.shift(1)).dropna()
        self.rv_daily = returns.rolling(window=5, min_periods=1).std() * np.sqrt(252) 
        self.rv_W = returns.rolling(window=5, min_periods=1).std() * np.sqrt(252)
        self.rv_M = returns.rolling(window=22, min_periods=5).std() * np.sqrt(252)
        self.realized_vol = pd.DataFrame({'rv': self.rv_daily, 'rv_W': self.rv_W, 'rv_M': self.rv_M}).dropna()
        self.returns = returns
        
    def fit_har(self):
        har_data = self.realized_vol.copy()
        X_har = pd.DataFrame({'rv_lag1': har_data['rv'].shift(1), 'rv_W_lag1': har_data['rv_W'].shift(1), 'rv_M_lag1': har_data['rv_M'].shift(1)}).dropna()
        y_har = har_data['rv'][X_har.index]
        self.har = LinearRegression()
        self.har.fit(X_har, y_har)
        self.X_har = X_har
        self.y_har = y_har
        
    def fit_garch(self):
        returns_clean = self.returns.dropna() * 10000  
        self.garch = arch_model(returns_clean, vol='Garch', p=1, q=1, rescale=False)
        self.garch_fit = self.garch.fit(disp='off')
        self.garch_volatility = self.garch_fit.conditional_volatility / 10000 * np.sqrt(252)
                
    def lstm_data(self):
        vol_data = self.realized_vol['rv'].dropna().values.reshape(-1, 1)
        vol_scaled = self.scaler.fit_transform(vol_data)
        X_lstm, y_lstm = [], []
        for i in range(self.lookback, len(vol_scaled)):
            X_lstm.append(vol_scaled[i-self.lookback:i, 0])
            y_lstm.append(vol_scaled[i, 0])
        self.X_lstm = np.array(X_lstm)
        self.y_lstm = np.array(y_lstm)
        self.X_lstm = self.X_lstm.reshape((self.X_lstm.shape[0], self.X_lstm.shape[1], 1))
        
    def fit_lstm(self):
        lstm_units = min(50, max(10, len(self.X_lstm) // 10))
        self.lstm = Sequential([
            Input(shape=(self.lookback, 1)),
            LSTM(lstm_units, return_sequences=True),
            Dropout(0.2),
            LSTM(lstm_units // 2, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        self.lstm.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        split_idx = max(1, int(0.8 * len(self.X_lstm)))
        X_train, X_val = self.X_lstm[:split_idx], self.X_lstm[split_idx:]
        y_train, y_val = self.y_lstm[:split_idx], self.y_lstm[split_idx:]
        if len(X_val) == 0:
            X_val, y_val = X_train, y_train
        epochs = min(10, max(10, len(X_train) // 5))
        self.lstm.fit(X_train, y_train, epochs=epochs, batch_size=min(32, max(1, len(X_train) // 10)), validation_data=(X_val, y_val), verbose=1)

    def fits(self):
        self.fit_har()
        self.fit_garch()
        self.lstm_data()
        self.fit_lstm()
        
    def predict(self, test_size=0.2):
        n_total = min(len(self.realized_vol), len(self.X_har), len(self.garch_volatility))
        n_test = max(5, min(int(test_size * n_total), n_total - 10))
        predictions = {}
        
        X_test_har = self.X_har.iloc[-n_test:]
        if len(X_test_har) > 0:
            predictions['HAR'] = self.har.predict(X_test_har)
            
        garch_aligned = self.garch_volatility.reindex(X_test_har.index, method='nearest')
        if len(garch_aligned.dropna()) > 0:
            predictions['GARCH'] = garch_aligned.dropna().values
            
        X_test_lstm = self.X_lstm[-n_test:]
        if len(X_test_lstm) > 0:
            pred_lstm_scaled = self.lstm.predict(X_test_lstm, verbose=0)
            predictions['LSTM'] = self.scaler.inverse_transform(pred_lstm_scaled).flatten()
            
        y = self.realized_vol['rv'].iloc[-n_test:].values
        min_len = min(len(pred) for pred in predictions.values())
        min_len = min(min_len, len(y))
        for key in predictions:
            predictions[key] = predictions[key][:min_len]
        y = y[:min_len]
        self.indiv_predictions = predictions
        self.y = y
        return predictions, y
        
    def fit(self):
        matrice = np.column_stack([self.indiv_predictions['HAR'], self.indiv_predictions['GARCH'], self.indiv_predictions['LSTM']])
        self.ensemble.fit(matrice, self.y)
        self.predictions = self.ensemble.predict(matrice)
        
    def evaluates(self):
        results = {}
        for name, pred in self.indiv_predictions.items():
            mse = mean_squared_error(self.y, pred)
            results[name] = {'MSE': mse, 'RMSE': np.sqrt(mse)}
        if self.predictions is not None:
            mse = mean_squared_error(self.y, self.predictions)
            results['Ensemble'] = {'MSE': mse, 'RMSE': np.sqrt(mse)}
        return pd.DataFrame(results).T
        
    def plot(self, figsize=(15, 10)):
        fig, axes = plt.subplots(4, 2, figsize=figsize)
        fig.suptitle('Prévisions de Volatilité', fontsize=16)
        x = range(len(self.y))
        
        axes[0,0].plot(x, self.y, label='RV')
        axes[0,0].plot(x, self.indiv_predictions['HAR'], label='HAR-RV')
        axes[0,0].set_title('Modèle HAR-RV')
        axes[0,0].legend()

        résidus_HAR = (self.indiv_predictions['HAR'] - self.y)
        tot_HAR = sum(abs(résidus_HAR))
        axes[0,1].plot(x, résidus_HAR, label=f'Résidus totaux: {tot_HAR}')        
        axes[0,1].set_title('Résidus modèle HAR-RV')
        axes[0,1].legend()
        
        axes[1,0].plot(x, self.y, label='RV')
        axes[1,0].plot(x, self.indiv_predictions['GARCH'], label='GARCH')
        axes[1,0].set_title('Modèle GARCH')
        axes[1,0].legend()
        
        résidus_GARCH = (self.indiv_predictions['GARCH'] - self.y)
        tot_GARCH = sum(abs(résidus_GARCH))
        axes[1,1].plot(x, résidus_GARCH, label=f'Résidus totaux:{tot_GARCH}')        
        axes[1,1].set_title('Résidus modèle GARCH')
        axes[1,1].legend()

        axes[2,0].plot(x, self.y, label='RV')
        axes[2,0].plot(x, self.indiv_predictions['LSTM'], label='LSTM')
        axes[2,0].set_title('Modèle LSTM')
        axes[2,0].legend()

        résidus_LSTM = (self.indiv_predictions['LSTM'] - self.y)
        tot_LSTM = sum(abs(résidus_LSTM))
        axes[2,1].plot(x, résidus_LSTM, label=f'Résidus totaux: {tot_LSTM}')        
        axes[2,1].set_title('Résidus modèle LSTM')
        axes[2,1].legend()
        
        axes[3,0].plot(x, self.y, label='RV')
        axes[3,0].plot(x, self.predictions, label='Modèle d\'Ensemble')
        axes[3,0].set_title('Modèle d\'Ensemble')
        axes[3,0].legend()

        résidus_E = (self.predictions - self.y)
        tot_E = sum(abs(résidus_E))
        axes[3,1].plot(x, résidus_E, label=f'Résidus totaux: {tot_E}')        
        axes[3,1].set_title('Résidus modèle Ensemble')
        axes[3,1].legend()

        plt.tight_layout()
        plt.show()
        
    def run(self, csv_path):
        self.load_data(csv_path)
        self.fits()
        predictions, y = self.predict()
        self.fit()
        results = self.evaluates()
        self.plot()
        return results, predictions

if __name__ == "__main__":
    ensemble = Vol(lookback=30, forecast_h=1)
    results, predictions = ensemble.run('dataset/eurusd-m1-bid-2019-01-01-2020-01-01.csv')
