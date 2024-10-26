import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sgp4.api import Satrec
from sgp4.api import jday
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class SatellitePredictor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = self._build_model()
        
    def _build_model(self):
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(24, 6), return_sequences=True),
            LSTM(32, activation='relu'),
            Dense(6)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def process_tle(self, line1, line2):
        satellite = Satrec.twoline2rv(line1, line2)
        positions = []
        
        for hour in range(24):
            jd, fr = jday(2024, 1, 1, hour, 0, 0)
            e, r, v = satellite.sgp4(jd, fr)
            
            if e != 0:
                raise ValueError("Error in SGP4 calculation")
                
            positions.append(r + v)
            
        return np.array(positions)
    
    def prepare_data(self, positions):
        scaled_data = self.scaler.fit_transform(positions)
        return np.expand_dims(scaled_data, axis=0)
    
    def predict_trajectory(self, line1, line2):
        positions = self.process_tle(line1, line2)
        X = self.prepare_data(positions)
        prediction = self.model.predict(X)
        return self.scaler.inverse_transform(prediction)

