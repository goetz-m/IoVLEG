import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np

class LCNN:
    """
    Lightweight CNN for DDoS Detection in IoV
    
    Architecture (from paper):
    - 2 Depthwise Separable Convolution blocks (each with Conv + ReLU + BatchNorm)
    - Global Average Pooling
    - Fully Connected layer with Softmax
    
    Input shape: (N, 12, 2) where N=5 (max packets), 12 features, 2 channels
    (packet-level and flow-level features)
    """
    
    def __init__(self, num_classes=6, max_packets=5, num_features=12, num_channels=2):
        """
        Initialize LCNN model
        
        Args:
            num_classes: Number of output classes (benign + attack types)
            max_packets: Maximum number of packets per flow (N parameter)
            num_features: Number of features per packet (12 as per paper)
            num_channels: Number of feature channels (2: packet-level and flow-level)
        """
        self.num_classes = num_classes
        self.max_packets = max_packets
        self.num_features = num_features
        self.num_channels = num_channels
        self.model = self._build_model()
    
    def _build_model(self):
        """Build the LCNN architecture"""
        
        # Input layer: (N x 12 x 2)
        inputs = layers.Input(shape=(self.max_packets, self.num_features, self.num_channels))
        
        # First Depthwise Separable Convolution Block
        # Depthwise convolution followed by pointwise convolution
        x = layers.SeparableConv2D(
            filters=16,
            kernel_size=(3, 3),
            padding='same',
            depthwise_initializer='he_normal',
            pointwise_initializer='he_normal'
        )(inputs)
        x = layers.ReLU()(x)
        x = layers.BatchNormalization()(x)
        
        # Second Depthwise Separable Convolution Block
        x = layers.SeparableConv2D(
            filters=32,
            kernel_size=(3, 3),
            padding='same',
            depthwise_initializer='he_normal',
            pointwise_initializer='he_normal'
        )(x)
        x = layers.ReLU()(x)
        x = layers.BatchNormalization()(x)
        
        # Global Average Pooling (replaces fully connected layer)
        x = layers.GlobalAveragePooling2D()(x)
        
        # Output layer with Softmax
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs, name='LCNN')
        
        return model