import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
import json
import logging
import warnings
import sys
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import requests
import hashlib
import joblib
from typing import Dict, List, Tuple, Optional, Union, Any
import asyncio
import aiohttp
from collections import Counter
import threading
import queue

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AIThreatDetector:
    """
    Advanced AI Security Platform for real-time threat detection
    """
    
    def __init__(self, model_type: str = "hybrid", config_path: str = "config.json"):
        """
        Initialize the AI Security Platform
        
        Args:
            model_type: Type of model to use ("cnn", "lstm", "hybrid", "ensemble")
            config_path: Path to configuration file
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.threshold = 0.85
        self.anomaly_scores = []
        self.prediction_history = []
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Database connection (simulated)
        self.db_connection = None
        self._init_database()
        
        # Real-time monitoring queue
        self.threat_queue = queue.Queue()
        self.monitoring_active = False
        
        logging.info(f"AI Security Platform initialized with {model_type} model")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        default_config = {
            "model_params": {
                "cnn_filters": 64,
                "cnn_kernel_size": 3,
                "lstm_units": 128,
                "dense_units": 64,
                "dropout_rate": 0.3,
                "learning_rate": 0.001
            },
            "training_params": {
                "epochs": 100,
                "batch_size": 32,
                "validation_split": 0.2,
                "patience": 10
            },
            "monitoring_params": {
                "check_interval": 5,
                "max_history": 1000,
                "alert_threshold": 0.9
            },
            "api_endpoints": {
                "threat_intel": "https://api.threatintel.example.com/v1/",
                "malware_analysis": "https://malware-api.example.com/analyze",
                "pattern_database": "https://patterns.example.com/api"
            }
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return default_config
    
    def _init_database(self):
        """Initialize database connection (simulated)"""
        # In production, this would connect to PostgreSQL
        self.db_connection = {
            'threats': [],
            'patterns': [],
            'logs': [],
            'users': []
        }
        logging.info("Database connection initialized")
    
    def prepare_data(self, data_path: str, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare and preprocess threat detection data
        
        Args:
            data_path: Path to CSV data file
            test_size: Proportion of data to use for testing
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        try:
            # Load data
            df = pd.read_csv(data_path)
            logging.info(f"Loaded data with shape: {df.shape}")
            
            # Separate features and labels
            X = df.drop('threat_label', axis=1).values
            y = df['threat_label'].values
            
            # Store feature names
            self.feature_names = df.drop('threat_label', axis=1).columns.tolist()
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Reshape for LSTM/CNN if needed
            if self.model_type in ["lstm", "hybrid"]:
                X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
            )
            
            logging.info(f"Data prepared: Train={X_train.shape}, Test={X_test.shape}")
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logging.error(f"Error preparing data: {e}")
            raise
    
    def build_model(self, input_shape: Tuple) -> tf.keras.Model:
        """
        Build the neural network model based on selected type
        
        Args:
            input_shape: Shape of input data
            
        Returns:
            Compiled TensorFlow model
        """
        model = Sequential()
        
        if self.model_type == "cnn":
            # CNN for spatial pattern detection
            model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())
            
        elif self.model_type == "lstm":
            # LSTM for temporal pattern detection
            model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape))
            model.add(Dropout(0.3))
            model.add(Bidirectional(LSTM(64)))
            
        elif self.model_type == "hybrid":
            # CNN-LSTM hybrid model
            model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Bidirectional(LSTM(128, return_sequences=True)))
            model.add(Dropout(0.3))
            model.add(Bidirectional(LSTM(64)))
            
        elif self.model_type == "ensemble":
            # Ensemble of multiple models (simplified)
            # In production, this would be a proper ensemble
            model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Bidirectional(LSTM(128, return_sequences=True)))
            model.add(Bidirectional(LSTM(64)))
            model.add(Flatten())
        
        # Common dense layers
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(len(self.label_encoder.classes_), activation='softmax'))
        
        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['model_params']['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        model.summary()
        self.model = model
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """
        Train the threat detection model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        try:
            # Create callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss' if X_val is not None else 'loss',
                    patience=self.config['training_params']['patience'],
                    restore_best_weights=True
                ),
                ModelCheckpoint(
                    'models/best_model.h5',
                    monitor='val_accuracy' if X_val is not None else 'accuracy',
                    save_best_only=True,
                    mode='max'
                )
            ]
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val) if X_val is not None else None,
                epochs=self.config['training_params']['epochs'],
                batch_size=self.config['training_params']['batch_size'],
                callbacks=callbacks,
                verbose=1
            )
            
            # Plot training history
            self._plot_training_history(history)
            
            # Save model and preprocessing objects
            self.save_model('models/threat_detector_full')
            
            logging.info("Model training completed successfully")
            return history
            
        except Exception as e:
            logging.error(f"Error during training: {e}")
            raise
    
    def predict(self, X: np.ndarray, return_proba: bool = False) -> Union[np.ndarray, Tuple]:
        """
        Predict threats on new data
        
        Args:
            X: Input features
            return_proba: Whether to return probability scores
            
        Returns:
            Predictions or (predictions, probabilities)
        """
        try:
            # Preprocess input
            X_scaled = self.scaler.transform(X)
            
            if self.model_type in ["lstm", "hybrid"]:
                X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
            
            # Make predictions
            probabilities = self.model.predict(X_scaled, verbose=0)
            predictions = np.argmax(probabilities, axis=1)
            
            # Decode labels
            decoded_predictions = self.label_encoder.inverse_transform(predictions)
            
            # Store in history
            for i, pred in enumerate(decoded_predictions):
                self.prediction_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'prediction': pred,
                    'confidence': float(np.max(probabilities[i])),
                    'features': X[i].tolist() if len(X) == 1 else 'batch'
                })
            
            # Keep history within limits
            if len(self.prediction_history) > self.config['monitoring_params']['max_history']:
                self.prediction_history = self.prediction_history[-self.config['monitoring_params']['max_history']:]
            
            if return_proba:
                return decoded_predictions, probabilities
            return decoded_predictions
            
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise
    
    def detect_anomalies(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Detect anomalous patterns in the data
        
        Args:
            X: Input features
            
        Returns:
            Dictionary with anomaly detection results
        """
        try:
            # Get prediction probabilities
            _, probabilities = self.predict(X, return_proba=True)
            
            # Calculate anomaly scores (1 - max probability)
            anomaly_scores = 1 - np.max(probabilities, axis=1)
            self.anomaly_scores.extend(anomaly_scores.tolist())
            
            # Identify anomalies above threshold
            anomalies = anomaly_scores > self.threshold
            anomaly_indices = np.where(anomalies)[0]
            
            # Prepare results
            results = {
                'total_samples': len(X),
                'anomalies_detected': int(np.sum(anomalies)),
                'anomaly_rate': float(np.mean(anomalies)),
                'anomaly_scores': anomaly_scores.tolist(),
                'anomaly_indices': anomaly_indices.tolist(),
                'confidence_scores': np.max(probabilities, axis=1).tolist(),
                'timestamp': datetime.now().isoformat()
            }
            
            # Log high-confidence anomalies
            high_confidence_anomalies = anomaly_scores > self.config['monitoring_params']['alert_threshold']
            if np.any(high_confidence_anomalies):
                logging.warning(f"High-confidence anomalies detected: {np.sum(high_confidence_anomalies)}")
                
                # Add to threat queue for real-time monitoring
                for idx in np.where(high_confidence_anomalies)[0]:
                    threat_info = {
                        'type': 'HIGH_CONFIDENCE_ANOMALY',
                        'score': float(anomaly_scores[idx]),
                        'features': X[idx].tolist() if len(X) == 1 else f"batch_index_{idx}",
                        'timestamp': datetime.now().isoformat()
                    }
                    self.threat_queue.put(threat_info)
            
            return results
            
        except Exception as e:
            logging.error(f"Error in anomaly detection: {e}")
            raise
    
    async def fetch_threat_intelligence(self, indicator: str) -> Dict[str, Any]:
        """
        Fetch threat intelligence from external APIs
        
        Args:
            indicator: IP, domain, hash, or other indicator
            
        Returns:
            Threat intelligence data
        """
        try:
            async with aiohttp.ClientSession() as session:
                # Query multiple threat intel sources (simulated)
                endpoints = self.config['api_endpoints']
                
                # Calculate hash if needed
                if len(indicator) == 64 or len(indicator) == 32:  # SHA256 or MD5
                    indicator_type = 'hash'
                elif '.' in indicator:
                    indicator_type = 'domain' if not indicator.replace('.', '').isdigit() else 'ip'
                else:
                    indicator_type = 'unknown'
                
                # Simulated API responses
                threat_data = {
                    'indicator': indicator,
                    'type': indicator_type,
                    'sources_checked': list(endpoints.keys()),
                    'malicious_score': np.random.uniform(0, 1),
                    'last_seen': datetime.now().isoformat(),
                    'tags': ['suspicious' if np.random.random() > 0.7 else 'clean'],
                    'raw_data': {}
                }
                
                # Simulate API calls
                for source, url in endpoints.items():
                    try:
                        # In production: await session.get(url + indicator)
                        threat_data['raw_data'][source] = {
                            'status': 'checked',
                            'result': 'malicious' if np.random.random() > 0.8 else 'clean'
                        }
                    except Exception as e:
                        threat_data['raw_data'][source] = {'error': str(e)}
                
                return threat_data
                
        except Exception as e:
            logging.error(f"Error fetching threat intelligence: {e}")
            return {'error': str(e)}
    
    def start_real_time_monitoring(self, data_stream, interval: int = 5):
        """
        Start real-time threat monitoring
        
        Args:
            data_stream: Source of real-time data
            interval: Check interval in seconds
        """
        self.monitoring_active = True
        
        def monitor():
            while self.monitoring_active:
                try:
                    # Get latest data from stream
                    latest_data = data_stream.get_latest()
                    
                    if latest_data is not None:
                        # Detect anomalies
                        results = self.detect_anomalies(latest_data)
                        
                        # Process threats in queue
                        self._process_threat_queue()
                        
                        # Log results
                        if results['anomalies_detected'] > 0:
                            logging.info(f"Real-time detection: {results['anomalies_detected']} anomalies found")
                    
                    # Sleep for interval
                    threading.Event().wait(interval)
                    
                except Exception as e:
                    logging.error(f"Monitoring error: {e}")
                    threading.Event().wait(interval)
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        logging.info(f"Real-time monitoring started with {interval}s interval")
    
    def _process_threat_queue(self):
        """Process threats in the queue"""
        while not self.threat_queue.empty():
            try:
                threat = self.threat_queue.get_nowait()
                
                # Log threat
                logging.warning(f"THREAT DETECTED: {threat}")
                
                # Store in database
                self.db_connection['threats'].append(threat)
                
                # Send alert (simulated)
                self._send_alert(threat)
                
                self.threat_queue.task_done()
                
            except queue.Empty:
                break
    
    def _send_alert(self, threat: Dict):
        """Send alert for detected threat"""
        alert_message = f"""
        🚨 SECURITY ALERT 🚨
        
        Threat Type: {threat.get('type', 'UNKNOWN')}
        Confidence Score: {threat.get('score', 0):.3f}
        Timestamp: {threat.get('timestamp', 'N/A')}
        
        Immediate action recommended.
        """
        
        # In production, send via email, Slack, webhook, etc.
        print(alert_message)
        
        # Log to file
        with open('alerts.log', 'a') as f:
            f.write(f"{datetime.now().isoformat()} - {alert_message}\n")
    
    def _plot_training_history(self, history):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Train')
        if 'val_accuracy' in history.history:
            axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].legend()
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Train')
        if 'val_loss' in history.history:
            axes[0, 1].plot(history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend()
        
        # Precision
        axes[1, 0].plot(history.history['precision'], label='Train')
        if 'val_precision' in history.history:
            axes[1, 0].plot(history.history['val_precision'], label='Validation')
        axes[1, 0].set_title('Precision')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].legend()
        
        # Recall
        axes[1, 1].plot(history.history['recall'], label='Train')
        if 'val_recall' in history.history:
            axes[1, 1].plot(history.history['val_recall'], label='Validation')
        axes[1, 1].set_title('Recall')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info("Training history plots saved")
    
    def save_model(self, path: str):
        """Save the complete model with preprocessing objects"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save TensorFlow model
        self.model.save(f'{path}_model.h5')
        
        # Save preprocessing objects
        with open(f'{path}_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(f'{path}_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save configuration
        with open(f'{path}_config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logging.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a saved model"""
        # Load TensorFlow model
        self.model = load_model(f'{path}_model.h5')
        
        # Load preprocessing objects
        with open(f'{path}_scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(f'{path}_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Load configuration
        with open(f'{path}_config.json', 'r') as f:
            self.config = json.load(f)
        
        logging.info(f"Model loaded from {path}")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive threat detection report"""
        report = {
            'platform_info': {
                'name': 'AI Security Platform',
                'version': '2.0.0',
                'model_type': self.model_type,
                'last_updated': datetime.now().isoformat()
            },
            'performance_metrics': {
                'total_predictions': len(self.prediction_history),
                'anomalies_detected': len([p for p in self.prediction_history 
                                          if p.get('confidence', 0) < (1 - self.threshold)]),
                'avg_confidence': np.mean([p.get('confidence', 0) for p in self.prediction_history]) 
                if self.prediction_history else 0
            },
            'threat_intelligence': {
                'total_threats': len(self.db_connection['threats']),
                'recent_threats': self.db_connection['threats'][-10:] if self.db_connection['threats'] else []
            },
            'recommendations': [
                'Review high-confidence anomalies',
                'Update threat intelligence feeds',
                'Consider model retraining if drift detected'
            ]
        }
        
        # Save report
        report_path = f'reports/threat_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        os.makedirs('reports', exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report


# Example usage and demonstration
class DemoDataStream:
    """Simulated data stream for demonstration"""
    
    def __init__(self, n_features: int = 50):
        self.n_features = n_features
        self.counter = 0
    
    def get_latest(self) -> np.ndarray:
        """Generate simulated network traffic data"""
        self.counter += 1
        
        # Generate normal data with occasional anomalies
        if self.counter % 100 == 0:
            # Anomaly pattern
            data = np.random.randn(1, self.n_features) * 3 + 10
        else:
            # Normal pattern
            data = np.random.randn(1, self.n_features)
        
        return data


def main():
    """Main demonstration function"""
    
    print("=" * 60)
    print("AI Security Platform - Advanced Threat Detection System")
    print("=" * 60)
    
    # Initialize platform
    security_platform = AIThreatDetector(model_type="hybrid")
    
    # For demo purposes, create sample data
    print("\n1. Creating sample threat detection dataset...")
    n_samples = 10000
    n_features = 50
    
    # Generate synthetic data
    X_demo = np.random.randn(n_samples, n_features)
    y_demo = np.random.choice(['normal', 'malware', 'intrusion', 'DDoS', 'phishing'], 
                              size=n_samples, p=[0.7, 0.1, 0.1, 0.05, 0.05])
    
    # Add some patterns
    X_demo[y_demo == 'malware'] += np.random.randn(np.sum(y_demo == 'malware'), n_features) * 2
    X_demo[y_demo == 'DDoS'] *= 3
    
    # Create DataFrame
    df_demo = pd.DataFrame(X_demo, columns=[f'feature_{i}' for i in range(n_features)])
    df_demo['threat_label'] = y_demo
    
    # Save sample data
    os.makedirs('data', exist_ok=True)
    df_demo.to_csv('data/threat_data_sample.csv', index=False)
    print(f"Sample data saved: {df_demo.shape}")
    
    # Prepare data
    print("\n2. Preparing data for training...")
    X_train, X_test, y_train, y_test = security_platform.prepare_data(
        'data/threat_data_sample.csv', test_size=0.3
    )
    
    # Build model
    print("\n3. Building neural network model...")
    input_shape = X_train.shape[1:] if security_platform.model_type == "cnn" else (X_train.shape[1], 1)
    security_platform.build_model(input_shape)
    
    # Train model
    print("\n4. Training threat detection model...")
    security_platform.train(X_train, y_train, X_test, y_test)
    
    # Test predictions
    print("\n5. Testing predictions...")
    sample_data = X_test[:5]
    predictions = security_platform.predict(sample_data)
    print(f"Sample predictions: {predictions}")
    
    # Anomaly detection
    print("\n6. Running anomaly detection...")
    anomaly_results = security_platform.detect_anomalies(sample_data)
    print(f"Anomalies detected: {anomaly_results['anomalies_detected']}")
    
    # Demonstrate real-time monitoring
    print("\n7. Starting real-time monitoring simulation...")
    data_stream = DemoDataStream(n_features=n_features)
    security_platform.start_real_time_monitoring(data_stream, interval=2)
    
    # Run monitoring for a short time
    print("Monitoring for 10 seconds (simulated)...")
    time.sleep(10)
    security_platform.monitoring_active = False
    
    # Generate report
    print("\n8. Generating threat detection report...")
    report = security_platform.generate_report()
    print(f"Report generated with {report['performance_metrics']['total_predictions']} predictions")
    
    # Demonstrate threat intelligence
    print("\n9. Fetching threat intelligence...")
    
    async def demo_threat_intel():
        indicators = ['8.8.8.8', 'malicious-domain.com', 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855']
        for indicator in indicators:
            intel = await security_platform.fetch_threat_intelligence(indicator)
            print(f"Indicator: {indicator} - Malicious score: {intel.get('malicious_score', 0):.3f}")
    
    asyncio.run(demo_threat_intel())
    
    print("\n" + "=" * 60)
    print("AI Security Platform demonstration completed!")
    print("Check the following files:")
    print("- models/best_model.h5 (trained model)")
    print("- training_history.png (training plots)")
    print("- alerts.log (security alerts)")
    print("- reports/ (threat detection reports)")
    print("=" * 60)


if __name__ == "__main__":
    main()
