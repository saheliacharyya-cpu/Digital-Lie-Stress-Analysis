import os
import numpy as np
from stress_detector import FacialStressDetector, VocalStressDetector, MultimodalFusionDetector

def test_facial_detector():
    print("Testing Facial Stress Detector...")
    detector = FacialStressDetector()
    assert detector.cnn_model is not None, "CNN model failed to load!"
    
    # Create dummy frame representing a face (BGR)
    dummy_frame = np.zeros((120, 120, 3), dtype=np.uint8) + 128
    
    # Run prediction (should return None since no actual face is detected, but ensures it runs without error)
    res = detector.analyze_frame(dummy_frame)
    print("Facial detector run completed. Result:", res)

def test_vocal_detector():
    print("Testing Vocal Stress Detector...")
    detector = VocalStressDetector()
    assert detector.voice_model is not None, "Voice model failed to load!"
    
    # Create dummy audio clip (16kHz, 1s duration)
    dummy_audio = np.random.normal(0, 0.05, 16000)
    import soundfile as sf
    temp_wav_path = "test_audio.wav"
    sf.write(temp_wav_path, dummy_audio, 16000)
    
    try:
        res = detector.analyze_audio_file(temp_wav_path)
        print("Vocal detector run completed. Result:", res)
        assert 'stress_score' in res, "Stress score missing in vocal prediction!"
    finally:
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)

def test_fusion_detector():
    print("Testing Multimodal Fusion Detector...")
    detector = MultimodalFusionDetector()
    
    face_res = {'stress_score': 0.8, 'emotion': 'anger', 'detected': True}
    voice_res = {'stress_score': 0.6}
    
    res = detector.fuse_predictions(face_res, voice_res)
    print("Fusion result:", res)
    expected = 0.45 * 0.8 + 0.55 * 0.6
    assert abs(res['fused_stress_score'] - expected) < 1e-5, "Fusion score math mismatch!"

if __name__ == '__main__':
    try:
        test_facial_detector()
        test_vocal_detector()
        test_fusion_detector()
        print("\nSUCCESS: All stress detectors validated and tests passed!")
    except AssertionError as e:
        print("\nFAILURE:", e)
        exit(1)
  