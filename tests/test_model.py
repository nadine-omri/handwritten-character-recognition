"""Tests unitaires"""

import torch
import unittest
from src.model import CNN, AdvancedCNN
import config

class TestModel(unittest.TestCase):
    """Tests du modèle"""
    
    def test_cnn_output_shape(self):
        """Tester la forme de sortie du CNN"""
        model = CNN()
        x = torch.randn(1, 1, 28, 28)
        output = model(x)
        self.assertEqual(output.shape, (1, config.NUM_CLASSES))
    
    def test_advanced_cnn_output_shape(self):
        """Tester la forme de sortie du CNN avancé"""
        model = AdvancedCNN()
        x = torch.randn(1, 1, 28, 28)
        output = model(x)
        self.assertEqual(output.shape, (1, config.NUM_CLASSES))
    
    def test_cnn_batch_processing(self):
        """Tester le traitement par batch"""
        model = CNN()
        x = torch.randn(32, 1, 28, 28)
        output = model(x)
        self.assertEqual(output.shape, (32, config.NUM_CLASSES))

if __name__ == "__main__":
    unittest.main()