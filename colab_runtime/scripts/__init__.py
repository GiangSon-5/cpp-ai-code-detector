# Colab Runtime — GPU Worker scripts package
"""
This package contains the GPU Worker (Dumb Worker) for C++ AI Code Detector.

Modules:
    config              — Device, paths, hyperparameters, logging
    engine              — ModelManager, RoBERTa Ensemble + LIG
    server              — FastAPI endpoints + ngrok + cache
    feature_extractor   — CppFeatureExtractorV8 (32 static features)
    hybrid_evaluator    — LightGBM prediction + Fusion score
"""
