from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

@dataclass
class PredictionMetrics:
    """Metrics for evaluating prediction accuracy"""
    position_rmse: float
    position_mae: float
    position_mse: float
    velocity_rmse: float
    velocity_mae: float
    velocity_mse: float
    evaluation_time: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            'position_metrics': {
                'rmse': self.position_rmse,
                'mae': self.position_mae,
                'mse': self.position_mse
            },
            'velocity_metrics': {
                'rmse': self.velocity_rmse,
                'mae': self.velocity_mae,
                'mse': self.velocity_mse
            },
            'evaluation_time': self.evaluation_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PredictionMetrics':
        pos_metrics = data.get('position_metrics', {})
        vel_metrics = data.get('velocity_metrics', {})
        
        return cls(
            position_rmse=pos_metrics.get('rmse', 0.0),
            position_mae=pos_metrics.get('mae', 0.0),
            position_mse=pos_metrics.get('mse', 0.0),
            velocity_rmse=vel_metrics.get('rmse', 0.0),
            velocity_mae=vel_metrics.get('mae', 0.0),
            velocity_mse=vel_metrics.get('mse', 0.0),
            evaluation_time=data.get('evaluation_time', datetime.utcnow().isoformat())
        )

@dataclass
class ConfidenceFactor:
    """Individual factor contributing to prediction confidence"""
    name: str
    value: float
    weight: float
    description: str
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'value': self.value,
            'weight': self.weight,
            'description': self.description
        }

@dataclass
class PredictionConfidence:
    """Confidence assessment for a prediction"""
    overall_confidence: float  # 0-1 scale
    confidence_level: str      # HIGH, MEDIUM, LOW, VERY_LOW
    factors: List[ConfidenceFactor] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'overall_confidence': self.overall_confidence,
            'confidence_level': self.confidence_level,
            'factors': [factor.to_dict() for factor in self.factors],
            'recommendations': self.recommendations
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PredictionConfidence':
        factors = [
            ConfidenceFactor(
                name=f['name'],
                value=f['value'],
                weight=f['weight'],
                description=f['description']
            ) for f in data.get('factors', [])
        ]
        
        return cls(
            overall_confidence=data['overall_confidence'],
            confidence_level=data['confidence_level'],
            factors=factors,
            recommendations=data.get('recommendations', [])
        )

@dataclass
class PredictionResult:
    """Result of a satellite trajectory prediction"""
    norad_id: str
    satellite_name: str
    prediction_time: str
    target_time: str
    hours_ahead: float
    
    # Predicted state
    position: List[float]  # [x, y, z] in km
    velocity: List[float]  # [vx, vy, vz] in km/s
    
    # Method information
    method: str  # 'sgp4', 'lstm', 'ensemble'
    model_version: Optional[str] = None
    
    # Quality metrics
    metrics: Optional[PredictionMetrics] = None
    confidence: Optional[PredictionConfidence] = None
    
    # Additional predictions (if available)
    sgp4_prediction: Optional[Dict] = None
    lstm_prediction: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        result = {
            'norad_id': self.norad_id,
            'satellite_name': self.satellite_name,
            'prediction_time': self.prediction_time,
            'target_time': self.target_time,
            'hours_ahead': self.hours_ahead,
            'prediction': {
                'position': self.position,
                'velocity': self.velocity
            },
            'method': self.method,
            'model_version': self.model_version
        }
        
        if self.metrics:
            result['metrics'] = self.metrics.to_dict()
        if self.confidence:
            result['confidence'] = self.confidence.to_dict()
        if self.sgp4_prediction:
            result['sgp4_prediction'] = self.sgp4_prediction
        if self.lstm_prediction:
            result['lstm_prediction'] = self.lstm_prediction
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PredictionResult':
        prediction_data = data.get('prediction', {})
        
        metrics = None
        if 'metrics' in data:
            metrics = PredictionMetrics.from_dict(data['metrics'])
        
        confidence = None
        if 'confidence' in data:
            confidence = PredictionConfidence.from_dict(data['confidence'])
        
        return cls(
            norad_id=data['norad_id'],
            satellite_name=data['satellite_name'],
            prediction_time=data['prediction_time'],
            target_time=data['target_time'],
            hours_ahead=data['hours_ahead'],
            position=prediction_data.get('position', [0, 0, 0]),
            velocity=prediction_data.get('velocity', [0, 0, 0]),
            method=data['method'],
            model_version=data.get('model_version'),
            metrics=metrics,
            confidence=confidence,
            sgp4_prediction=data.get('sgp4_prediction'),
            lstm_prediction=data.get('lstm_prediction')
        )

@dataclass
class BatchPredictionRequest:
    """Request for batch prediction of multiple satellites"""
    satellites: List[Dict]  # List of satellite requests
    common_parameters: Dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            'satellites': self.satellites,
            'common_parameters': self.common_parameters,
            'timestamp': self.timestamp,
            'count': len(self.satellites)
        }

@dataclass
class BatchPredictionResult:
    """Result of batch prediction"""
    request_id: str
    predictions: Dict[str, PredictionResult]
    successful_count: int
    failed_count: int
    processing_time: float  # seconds
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            'request_id': self.request_id,
            'predictions': {k: v.to_dict() for k, v in self.predictions.items()},
            'successful_count': self.successful_count,
            'failed_count': self.failed_count,
            'processing_time': self.processing_time,
            'timestamp': self.timestamp,
            'total_requested': self.successful_count + self.failed_count
        }

@dataclass
class ModelPerformance:
    """Performance metrics for prediction models"""
    model_name: str
    test_period: str
    sample_count: int
    
    # Accuracy metrics
    mean_position_error: float  # km
    mean_velocity_error: float  # km/s
    max_position_error: float   # km
    max_velocity_error: float   # km/s
    
    # Time-based performance
    short_term_accuracy: float  # 1-6 hours
    medium_term_accuracy: float # 6-24 hours
    long_term_accuracy: float   # 24+ hours
    
    # Computational metrics
    prediction_time_ms: float
    memory_usage_mb: float
    
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            'model_name': self.model_name,
            'test_period': self.test_period,
            'sample_count': self.sample_count,
            'accuracy_metrics': {
                'mean_position_error_km': self.mean_position_error,
                'mean_velocity_error_km_s': self.mean_velocity_error,
                'max_position_error_km': self.max_position_error,
                'max_velocity_error_km_s': self.max_velocity_error
            },
            'time_based_performance': {
                'short_term_1_6h': self.short_term_accuracy,
                'medium_term_6_24h': self.medium_term_accuracy,
                'long_term_24h_plus': self.long_term_accuracy
            },
            'computational_metrics': {
                'prediction_time_ms': self.prediction_time_ms,
                'memory_usage_mb': self.memory_usage_mb
            },
            'last_updated': self.last_updated
        }

@dataclass
class TrainingData:
    """Training data for machine learning models"""
    satellite_id: str
    time_series: List[List[float]]  # Time series of [x,y,z,vx,vy,vz]
    timestamps: List[str]
    data_quality: float  # 0-1 quality score
    source: str = "sgp4"
    
    def to_dict(self) -> Dict:
        return {
            'satellite_id': self.satellite_id,
            'time_series': self.time_series,
            'timestamps': self.timestamps,
            'data_quality': self.data_quality,
            'source': self.source,
            'sample_count': len(self.time_series)
        }
    
    def get_numpy_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get numpy arrays for training"""
        X = np.array(self.time_series[:-1])  # Input sequences
        y = np.array(self.time_series[1:])   # Target sequences
        return X, y

@dataclass
class ModelConfiguration:
    """Configuration for prediction models"""
    model_type: str  # 'lstm', 'transformer', 'hybrid'
    parameters: Dict[str, Any]
    training_config: Dict[str, Any]
    version: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            'model_type': self.model_type,
            'parameters': self.parameters,
            'training_config': self.training_config,
            'version': self.version,
            'created_at': self.created_at
        }

class PredictionHistory:
    """Track prediction history for analysis"""
    
    def __init__(self):
        self.predictions: List[PredictionResult] = []
        self.performance_metrics: Dict[str, ModelPerformance] = {}
    
    def add_prediction(self, prediction: PredictionResult):
        """Add a prediction to history"""
        self.predictions.append(prediction)
        
        # Keep only recent predictions (last 1000)
        if len(self.predictions) > 1000:
            self.predictions = self.predictions[-1000:]
    
    def get_model_accuracy(self, model_name: str, time_window_hours: int = 24) -> Optional[float]:
        """Get average accuracy for a model over time window"""
        cutoff_time = datetime.utcnow() - datetime.timedelta(hours=time_window_hours)
        
        relevant_predictions = [
            p for p in self.predictions 
            if (p.method == model_name and 
                datetime.fromisoformat(p.prediction_time.replace('Z', '')) > cutoff_time and
                p.metrics is not None)
        ]
        
        if not relevant_predictions:
            return None
        
        # Average position RMSE
        rmse_values = [p.metrics.position_rmse for p in relevant_predictions]
        return sum(rmse_values) / len(rmse_values)
    
    def get_statistics(self) -> Dict:
        """Get prediction statistics"""
        if not self.predictions:
            return {'total_predictions': 0}
        
        method_counts = {}
        total_accuracy = 0
        accuracy_count = 0
        
        for pred in self.predictions:
            method_counts[pred.method] = method_counts.get(pred.method, 0) + 1
            
            if pred.metrics:
                total_accuracy += pred.metrics.position_rmse
                accuracy_count += 1
        
        return {
            'total_predictions': len(self.predictions),
            'methods_used': method_counts,
            'average_position_rmse': total_accuracy / accuracy_count if accuracy_count > 0 else None,
            'recent_predictions': len([p for p in self.predictions 
                                     if datetime.fromisoformat(p.prediction_time.replace('Z', '')) > 
                                     datetime.utcnow() - datetime.timedelta(hours=24)])
        }

# Global prediction history instance
prediction_history = PredictionHistory()