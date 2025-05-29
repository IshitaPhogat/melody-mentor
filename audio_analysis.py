import librosa
import numpy as np
import scipy

def load_audio(audio_path):
    # Loads and resamples the audio file.
    try:
        audio, sr = librosa.load(audio_path)
        return audio, sr
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None, None

def extract_features(audio, sr):
    # Extracts audio features relevant to singing quality.
    features = {}
    
    # Extract pitch using PYIN (more reliable for singing voice)
    f0, voiced_flag, voiced_probs = librosa.pyin(audio, 
                                               fmin=librosa.note_to_hz('C2'), 
                                               fmax=librosa.note_to_hz('C7'),
                                               sr=sr)
    # Replace NaN values with zeros
    f0 = np.nan_to_num(f0)
    features['pitch'] = f0
    
    # RMS energy (volume)
    features['rms'] = librosa.feature.rms(y=audio)[0]
    
    # Spectral centroid (brightness/timbre)
    features['spectral_centroid'] = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    
    return features

def compare_features(ref_features, user_features):
    # Compares the features of the reference and user audio.
    comparison = {}
    
    # Ensure consistent length for comparison
    min_length = min(len(ref_features['pitch']), len(user_features['pitch']))
    
    # Compare pitch
    ref_pitch = ref_features['pitch'][:min_length]
    user_pitch = user_features['pitch'][:min_length]
    
    # Filter out zeros (unvoiced segments)
    voiced_indices = (ref_pitch > 0) & (user_pitch > 0)
    
    if np.any(voiced_indices):
        # Calculate pitch deviation in cents (musical unit) for better feedback
        ref_pitch_voiced = ref_pitch[voiced_indices]
        user_pitch_voiced = user_pitch[voiced_indices]
        
        # Convert Hz difference to cents (musical perception)
        cents_diff = 1200 * np.log2(user_pitch_voiced / ref_pitch_voiced)
        comparison['pitch_deviation'] = np.mean(np.abs(cents_diff))
    else:
        comparison['pitch_deviation'] = None
    
    # Compare RMS energy (volume)
    min_rms_length = min(len(ref_features['rms']), len(user_features['rms']))
    comparison['rms_deviation'] = np.mean(np.abs(
        ref_features['rms'][:min_rms_length] - user_features['rms'][:min_rms_length]
    ))
    
    # Compare spectral centroid (timbre)
    min_spec_length = min(len(ref_features['spectral_centroid']), len(user_features['spectral_centroid']))
    comparison['spectral_centroid_deviation'] = np.mean(np.abs(
        ref_features['spectral_centroid'][:min_spec_length] - user_features['spectral_centroid'][:min_spec_length]
    ))
    
    return comparison

def give_feedback(comparison_results):
    # Provides feedback to the user based on the comparison.
    feedback = []
    
    if comparison_results['pitch_deviation'] is not None:
        # Pitch feedback based on cents difference
        if comparison_results['pitch_deviation'] < 50:  # Less than 50 cents (half semitone)
            feedback.append("Your pitch accuracy is excellent! You're staying very close to the original melody.")
        elif comparison_results['pitch_deviation'] < 100:  # Less than 1 semitone
            feedback.append("Your pitch is good but could use some fine-tuning. Try focusing on the more challenging note transitions.")
        elif comparison_results['pitch_deviation'] < 200:  # Less than 2 semitones
            feedback.append("Your pitch needs some work. Try singing with the reference audio and pay attention to the melody.")
        else:
            feedback.append("Your pitch needs significant improvement. Consider practicing with a piano or vocal warm-ups to improve your pitch accuracy.")
    else:
        feedback.append("Could not compare pitch. Ensure both audios have clear vocal content.")
    
    # Volume feedback
    if comparison_results['rms_deviation'] < 0.05:
        feedback.append("Your volume control is excellent!")
    elif comparison_results['rms_deviation'] < 0.1:
        feedback.append("Your volume consistency is good. Minor adjustments needed for perfect dynamics.")
    else:
        feedback.append("Work on maintaining more consistent volume. Practice breath control for better volume regulation.")
    
    # Timbre feedback
    if comparison_results['spectral_centroid_deviation'] < 200:
        feedback.append("Your vocal tone/timbre matches the original very well!")
    elif comparison_results['spectral_centroid_deviation'] < 500:
        feedback.append("Your vocal tone is fairly close to the original. Focus on vowel shapes to better match the timbre.")
    else:
        feedback.append("Your vocal timbre differs significantly from the reference. Experiment with different vocal techniques and resonance to match the original sound better.")
    
    return feedback