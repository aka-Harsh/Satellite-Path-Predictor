from flask import Flask, render_template, request, jsonify
from model import SatellitePredictor
from utils import validate_tle, format_prediction
import numpy as np
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
import base64

app = Flask(__name__)
predictor = SatellitePredictor()


class SatelliteEvaluator:
    def __init__(self, predictor):
        self.predictor = predictor
        
    def generate_true_trajectory(self, line1, line2, hours=48):
        satellite = Satrec.twoline2rv(line1, line2)
        positions = []
        timestamps = []
        
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        for i in range(hours):
            current_time = start_time + timedelta(hours=i)
            jd, fr = jday(current_time.year, current_time.month, current_time.day,
                         current_time.hour, current_time.minute, current_time.second)
            
            e, r, v = satellite.sgp4(jd, fr)
            if e == 0:
                positions.append(r + v)
                timestamps.append(current_time)
                
        return np.array(positions), timestamps
    
    def evaluate_predictions(self, line1, line2):
        """Fixed evaluation method"""
        true_positions, _ = self.generate_true_trajectory(line1, line2)
        predicted_positions = self.predictor.predict_trajectory(line1, line2)
        
        true_first_point = true_positions[0]
        
        position_mse = mean_squared_error(true_first_point[:3].reshape(1, -1), 
                                        predicted_positions[0, :3].reshape(1, -1))
        position_mae = mean_absolute_error(true_first_point[:3].reshape(1, -1), 
                                         predicted_positions[0, :3].reshape(1, -1))
        position_rmse = np.sqrt(position_mse)
        
        velocity_mse = mean_squared_error(true_first_point[3:].reshape(1, -1), 
                                        predicted_positions[0, 3:].reshape(1, -1))
        velocity_mae = mean_absolute_error(true_first_point[3:].reshape(1, -1), 
                                         predicted_positions[0, 3:].reshape(1, -1))
        velocity_rmse = np.sqrt(velocity_mse)
        
        return {
            'position_metrics': {
                'mse': float(position_mse),
                'mae': float(position_mae),
                'rmse': float(position_rmse)
            },
            'velocity_metrics': {
                'mse': float(velocity_mse),
                'mae': float(velocity_mae),
                'rmse': float(velocity_rmse)
            }
        }
    
    def plot_trajectory(self, line1, line2):
        true_positions, _ = self.generate_true_trajectory(line1, line2, hours=24)  # Reduced to 24 hours for better visualization
        predicted = self.predictor.predict_trajectory(line1, line2)
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot(true_positions[:, 0], true_positions[:, 1], true_positions[:, 2], 
                'b-', label='True Trajectory', alpha=0.6)
        
        ax.scatter(predicted[0, 0], predicted[0, 1], predicted[0, 2], 
                  color='red', s=100, label='Predicted Position')
        
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        earth_radius = 6371  # km
        x = earth_radius * np.outer(np.cos(u), np.sin(v))
        y = earth_radius * np.outer(np.sin(u), np.sin(v))
        z = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='lightblue', alpha=0.1)
        
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        ax.set_title('Satellite Trajectory: True vs Predicted')
        ax.legend()
        
        ax.grid(True)
        ax.view_init(elev=20, azim=45)
        
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return plot_url

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    line1 = data.get('line1')
    line2 = data.get('line2')
    
    is_valid, error_message = validate_tle(line1, line2)
    if not is_valid:
        return jsonify({'error': error_message}), 400
    
    try:
        prediction = predictor.predict_trajectory(line1, line2)
        return jsonify(format_prediction(prediction))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.get_json()
    line1 = data.get('line1')
    line2 = data.get('line2')
    
    is_valid, error_message = validate_tle(line1, line2)
    if not is_valid:
        return jsonify({'error': error_message}), 400
    
    try:
        evaluator = SatelliteEvaluator(predictor)
        metrics = evaluator.evaluate_predictions(line1, line2)
        plot_url = evaluator.plot_trajectory(line1, line2)
        
        return jsonify({
            'metrics': metrics,
            'prediction': format_prediction(predictor.predict_trajectory(line1, line2)),
            'plot': plot_url
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

