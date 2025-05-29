import streamlit as st
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import os
from audio_analysis import load_audio, extract_features, compare_features, give_feedback

class Frontend:
    def __init__(self):
        st.set_page_config(page_title="Melody Mentor")
        st.title("Learn how to sing better")
        st.success("Welcome to the application. Here people from any age group can learn how to sing and improve their singing capabilities")
        st.write("The application uses a reference audio file of the song you are trying to learn how to sing. On the basis of different parameters which determine good singing, the user is provided with inputs on how to improve their singing")
        self.ref_audio_file = None
        self.user_audio_file = None
        self.user_uploaded_file = None
        self.input_method = None
        self.refFile()
        self.inputMethod()

    def refFile(self):
        # File uploader for reference audio
        self.ref_audio_file = st.file_uploader(
            label="Upload your reference audio file",
            type=["wav", "mp3"],
            accept_multiple_files=False,
            key="ref_file_uploader"
        )

    def inputMethod(self):
        # Let the user choose how to provide their singing sample
        st.subheader("Your Singing Sample")
        self.input_method = st.radio(
            "Choose how to provide your singing sample:",
            ["Record Audio", "Upload Audio File"],
            key="input_method"
        )
        
        if self.input_method == "Record Audio":
            # Audio recorder for user's singing
            self.user_audio_file = st.audio_input("Record yourself singing the song", key="user_audio_input")
        else:
            # File uploader for user's singing
            self.user_uploaded_file = st.file_uploader(
                label="Upload your singing audio file",
                type=["wav", "mp3"],
                accept_multiple_files=False,
                key="user_file_uploader"
            )

    def run(self):
        # Check if both files are available
        analyze_button = st.button("Analyze my singing")
        
        if analyze_button and self.ref_audio_file:
            # Check that user provided some form of audio input
            user_has_input = (self.input_method == "Record Audio" and self.user_audio_file) or \
                             (self.input_method == "Upload Audio File" and self.user_uploaded_file)
                             
            if not user_has_input:
                st.error("Please provide your singing sample before analysis.")
                return
                
            with st.spinner("Analyzing your singing..."):
                # Save uploaded reference file to temporary path
                ref_audio_path = f"temp_ref.{self.ref_audio_file.name.split('.')[-1]}"
                user_audio_path = "temp_user.wav"
                
                # Write the reference file using getbuffer()
                with open(ref_audio_path, "wb") as f:
                    f.write(self.ref_audio_file.getbuffer())
                
                # Handle user audio based on input method
                if self.input_method == "Record Audio":
                    # For audio_input, the data is directly bytes
                    with open(user_audio_path, "wb") as f:
                        f.write(self.user_audio_file.getvalue())
                else:
                    # For uploaded file, use getbuffer()
                    user_audio_path = f"temp_user.{self.user_uploaded_file.name.split('.')[-1]}"
                    with open(user_audio_path, "wb") as f:
                        f.write(self.user_uploaded_file.getbuffer())

                # Load and process audio using functions from audio_analysis.py
                ref_audio, ref_sr = load_audio(ref_audio_path)
                user_audio, user_sr = load_audio(user_audio_path)

                if ref_audio is not None and user_audio is not None:
                    # Resample user audio to reference audio's sampling rate if needed
                    if ref_sr != user_sr:
                        user_audio = librosa.resample(user_audio, orig_sr=user_sr, target_sr=ref_sr)
                        user_sr = ref_sr

                    # Extract features
                    ref_features = extract_features(ref_audio, ref_sr)
                    user_features = extract_features(user_audio, user_sr)
                    
                    # Compare and generate feedback
                    comparison_results = compare_features(ref_features, user_features)
                    feedback = give_feedback(comparison_results)

                    # Display visualizations
                    self.plot_audio_features(ref_audio, ref_sr, user_audio, user_sr, ref_features, user_features)

                    # Display feedback
                    st.subheader("Feedback on Your Singing:")
                    for fb in feedback:
                        st.info(fb)
                    
                    # Display comparison metrics
                    st.subheader("Technical Metrics:")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if comparison_results['pitch_deviation'] is not None:
                            st.metric(label="Pitch Deviation (cents)", 
                                     value=f"{comparison_results['pitch_deviation']:.1f}")
                        else:
                            st.metric(label="Pitch Deviation", value="N/A")
                    with col2:
                        st.metric(label="Volume Consistency", 
                                 value=f"{comparison_results['rms_deviation']:.3f}")
                    with col3:
                        st.metric(label="Timbre Match", 
                                 value=f"{comparison_results['spectral_centroid_deviation']:.1f}")
                    
                    # Let user listen to both audios for comparison
                    st.subheader("Listen and Compare:")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.audio(ref_audio_path, format=f"audio/{ref_audio_path.split('.')[-1]}")
                        st.caption("Reference Audio")
                    with col2:
                        st.audio(user_audio_path, format=f"audio/{user_audio_path.split('.')[-1]}")
                        st.caption("Your Singing")
                else:
                    st.error("Failed to process audio files. Please check file formats and try again.")
                
                # Clean up temporary files
                try:
                    os.remove(ref_audio_path)
                    os.remove(user_audio_path)
                except:
                    pass

            st.balloons()

    def plot_audio_features(self, ref_audio, ref_sr, user_audio, user_sr, ref_features, user_features):
        """
        Plots audio features for visual comparison.
        """
        # Create figure for visualizations
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        
        # Plot waveforms
        axes[0].set_title('Audio Waveforms')
        librosa.display.waveshow(ref_audio, sr=ref_sr, alpha=0.6, ax=axes[0], label='Reference')
        librosa.display.waveshow(user_audio, sr=user_sr, color='r', alpha=0.6, ax=axes[0], label='Your Singing')
        axes[0].legend(loc='upper right')
        
        # Plot pitch contours
        # Get time bases for both signals
        ref_times = librosa.times_like(ref_features['pitch'], sr=ref_sr)
        user_times = librosa.times_like(user_features['pitch'], sr=user_sr)
        
        # Truncate to the same length for visualization
        min_len = min(len(ref_times), len(user_times))
        ref_times = ref_times[:min_len]
        user_times = user_times[:min_len]
        ref_pitch = ref_features['pitch'][:min_len]
        user_pitch = user_features['pitch'][:min_len]
        
        axes[1].set_title('Pitch Contours')
        axes[1].plot(ref_times, ref_pitch, label='Reference Pitch', alpha=0.7)
        axes[1].plot(user_times, user_pitch, label='Your Pitch', color='r', alpha=0.7)
        axes[1].set_ylabel('Frequency (Hz)')
        axes[1].legend(loc='upper right')
        
        # Plot volume (RMS energy)
        # Truncate to the same length for visualization
        min_rms_len = min(len(ref_features['rms']), len(user_features['rms']))
        ref_rms_times = librosa.times_like(ref_features['rms'], sr=ref_sr)[:min_rms_len]
        user_rms_times = librosa.times_like(user_features['rms'], sr=user_sr)[:min_rms_len]
        
        axes[2].set_title('Volume (RMS Energy)')
        axes[2].plot(ref_rms_times, ref_features['rms'][:min_rms_len], label='Reference Volume', alpha=0.7)
        axes[2].plot(user_rms_times, user_features['rms'][:min_rms_len], label='Your Volume', color='r', alpha=0.7)
        axes[2].set_ylabel('RMS Energy')
        axes[2].set_xlabel('Time (seconds)')
        axes[2].legend(loc='upper right')
        
        plt.tight_layout()
        st.pyplot(fig)

def main():
    frontend = Frontend()
    frontend.run()


if __name__ == "__main__":
    main()