import streamlit as st

st.set_page_config(page_title="Melody Mentor", layout="wide")

import collections.abc

if not hasattr(collections.abc, 'MutableMapping'):
    collections.abc.MutableMapping = collections.abc.MutableMapping
if not hasattr(collections.abc, 'Mapping'):
    collections.abc.Mapping = collections.abc.Mapping
if not hasattr(collections.abc, 'Iterable'):
    collections.abc.Iterable = collections.abc.Iterable
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
from datetime import datetime
from audio_analysis import load_audio, extract_features, compare_features, give_feedback

# Firebase imports - Only Admin SDK
import firebase_admin
from firebase_admin import credentials, firestore, auth as firebase_auth
import requests

# Firebase Configuration for Web API
FIREBASE_WEB_API_KEY = "AIzaSyB8roQ3y_BI-FCHySoTgmGYo8CJo_RZP3c"
FIREBASE_AUTH_URL = "https://identitytoolkit.googleapis.com/v1/accounts"


# Initialize Firebase Admin
@st.cache_resource
def init_firebase():
    try:
        # Initialize Firebase Admin
        if not firebase_admin._apps:
            import json
            service_account_info = json.loads(st.secrets["firebase_service_account_key"])
            cred = credentials.Certificate(service_account_info)
            firebase_admin.initialize_app(cred)

        db = firestore.client()
        return db
    except Exception as e:
        st.error(f"Firebase initialization error: {e}")
        return None


db = init_firebase()


class AuthHandler:
    def __init__(self):
        self.api_key = FIREBASE_WEB_API_KEY
        self.auth_url = FIREBASE_AUTH_URL

    def sign_up(self, email, password, name):
        """Create new user using Firebase REST API"""
        try:
            # Create user with Firebase Auth REST API
            url = f"{self.auth_url}:signUp?key={self.api_key}"
            payload = {
                "email": email,
                "password": password,
                "returnSecureToken": True
            }

            response = requests.post(url, json=payload)

            if response.status_code == 200:
                user_data = response.json()
                user_id = user_data['localId']

                # Create user profile in Firestore
                if db:
                    user_ref = db.collection('users').document(user_id)
                    user_ref.set({
                        'name': name,
                        'email': email,
                        'created_at': datetime.now(),
                        'total_analyses': 0
                    })

                return user_data, None
            else:
                error_data = response.json()
                error_message = error_data.get('error', {}).get('message', 'Unknown error')
                return None, self._format_error_message(error_message)

        except Exception as e:
            return None, str(e)

    def sign_in(self, email, password):
        """Sign in user using Firebase REST API"""
        try:
            url = f"{self.auth_url}:signInWithPassword?key={self.api_key}"
            payload = {
                "email": email,
                "password": password,
                "returnSecureToken": True
            }

            response = requests.post(url, json=payload)

            if response.status_code == 200:
                user_data = response.json()
                user_id = user_data['localId']

                # Get additional user data from Firestore
                if db:
                    user_doc = db.collection('users').document(user_id).get()
                    if user_doc.exists:
                        firestore_data = user_doc.to_dict()
                        user_data.update(firestore_data)

                return user_data, None
            else:
                error_data = response.json()
                error_message = error_data.get('error', {}).get('message', 'Unknown error')
                return None, self._format_error_message(error_message)

        except Exception as e:
            return None, str(e)

    def verify_token(self, id_token):
        """Verify Firebase ID token using Admin SDK"""
        try:
            decoded_token = firebase_auth.verify_id_token(id_token)
            return decoded_token, None
        except Exception as e:
            return None, str(e)

    def _format_error_message(self, error_message):
        """Format Firebase error messages to be user-friendly"""
        error_mapping = {
            'EMAIL_EXISTS': 'This email is already registered. Please use a different email or try logging in.',
            'OPERATION_NOT_ALLOWED': 'Email/password accounts are not enabled. Please contact support.',
            'TOO_MANY_ATTEMPTS_TRY_LATER': 'Too many unsuccessful attempts. Please try again later.',
            'EMAIL_NOT_FOUND': 'No account found with this email address.',
            'INVALID_PASSWORD': 'Incorrect password. Please try again.',
            'USER_DISABLED': 'This account has been disabled. Please contact support.',
            'INVALID_EMAIL': 'Please enter a valid email address.',
            'WEAK_PASSWORD': 'Password should be at least 6 characters long.'
        }
        return error_mapping.get(error_message, f"Authentication error: {error_message}")


class Frontend:
    def __init__(self):
        # Initialize session state
        if 'logged_in' not in st.session_state:
            st.session_state.logged_in = False
        if 'user' not in st.session_state:
            st.session_state.user = None
        if 'id_token' not in st.session_state:
            st.session_state.id_token = None

        # Initialize auth handler
        self.auth_handler = AuthHandler()

        # Check authentication status
        if not st.session_state.logged_in:
            self.show_auth_page()
        else:
            self.show_main_app()

    def show_auth_page(self):
        """Display authentication page"""
        st.title("🎵 Melody Mentor")
        st.success(
            "Welcome to the application. Here people from any age group can learn how to sing and improve their singing capabilities")

        # Create tabs for login and signup
        tab1, tab2 = st.tabs(["Login", "Sign Up"])

        with tab1:
            self.show_login_form()

        with tab2:
            self.show_signup_form()

    def show_login_form(self):
        """Display login form"""
        st.subheader("Login to Your Account")

        with st.form("login_form"):
            email = st.text_input("📧 Email")
            password = st.text_input("🔒 Password", type="password")
            login_button = st.form_submit_button("Login", use_container_width=True)

            if login_button:
                if email and password:
                    with st.spinner("Logging you in..."):
                        user, error = self.auth_handler.sign_in(email, password)
                        if user:
                            st.session_state.user = user
                            st.session_state.logged_in = True
                            st.session_state.id_token = user.get('idToken')
                            st.success("Login successful!")
                            st.rerun()
                        else:
                            st.error(error)
                else:
                    st.error("Please fill in all fields")

    def show_signup_form(self):
        """Display signup form"""
        st.subheader("Create New Account")

        with st.form("signup_form"):
            name = st.text_input("👤 Full Name")
            email = st.text_input("📧 Email")
            password = st.text_input("🔒 Password", type="password")
            confirm_password = st.text_input("🔒 Confirm Password", type="password")
            signup_button = st.form_submit_button("Sign Up", use_container_width=True)

            if signup_button:
                if name and email and password and confirm_password:
                    if password == confirm_password:
                        if len(password) >= 6:
                            with st.spinner("Creating your account..."):
                                user, error = self.auth_handler.sign_up(email, password, name)
                                if user:
                                    st.success("Account created successfully! Please login.")
                                else:
                                    st.error(error)
                        else:
                            st.error("Password must be at least 6 characters")
                    else:
                        st.error("Passwords don't match")
                else:
                    st.error("Please fill in all fields")

    def show_main_app(self):
        """Main application after authentication"""
        # Header with user info and logout
        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            st.title("🎵 Melody Mentor")

        with col2:
            user_name = st.session_state.user.get('name', 'User')
            st.write(f"Welcome, {user_name}!")

        with col3:
            if st.button("Logout", use_container_width=True):
                self.logout()

        # Sidebar navigation
        with st.sidebar:
            st.header("🎵 Navigation")
            page = st.selectbox(
                "Choose a page:",
                ["🎤 Analyze Singing", "📊 My History", "👤 Profile"],
                key="navigation"
            )

        # Page routing
        if page == "🎤 Analyze Singing":
            self.show_analysis_page()
        elif page == "📊 My History":
            self.show_history_page()
        elif page == "👤 Profile":
            self.show_profile_page()

    def show_analysis_page(self):
        """Your existing audio analysis functionality"""
        st.write(
            "The application uses a reference audio file of the song you are trying to learn how to sing. On the basis of different parameters which determine good singing, the user is provided with inputs on how to improve their singing")

        self.ref_audio_file = None
        self.user_audio_file = None
        self.user_uploaded_file = None
        self.input_method = None

        # Initialize analysis components
        self.refFile()
        self.inputMethod()
        self.run_analysis()

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

    def run_analysis(self):
        # Check if both files are available
        analyze_button = st.button("🎯 Analyze my singing", use_container_width=True)

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

                    # Save analysis to Firestore
                    self.save_analysis_to_firestore(comparison_results, self.ref_audio_file.name)

                    # Display visualizations
                    self.plot_audio_features(ref_audio, ref_sr, user_audio, user_sr, ref_features, user_features)

                    # Display feedback
                    st.subheader("🎯 Feedback on Your Singing:")
                    for fb in feedback:
                        st.info(fb)

                    # Display comparison metrics
                    st.subheader("📊 Technical Metrics:")
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
                    st.subheader("🎧 Listen and Compare:")
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

    def save_analysis_to_firestore(self, comparison_results, ref_file_name):
        """Save analysis results to Firestore"""
        if not db or not st.session_state.user:
            return

        try:
            user_id = st.session_state.user['localId']

            # Prepare analysis data
            analysis_data = {
                'reference_file_name': ref_file_name,
                # Convert np.float32 to standard Python float
                'pitch_deviation': float(comparison_results.get('pitch_deviation')) if comparison_results.get(
                    'pitch_deviation') is not None else None,
                'rms_deviation': float(comparison_results.get('rms_deviation')) if comparison_results.get(
                    'rms_deviation') is not None else None,
                'spectral_centroid_deviation': float(
                    comparison_results.get('spectral_centroid_deviation')) if comparison_results.get(
                    'spectral_centroid_deviation') is not None else None,
                'timestamp': datetime.now(),
                'input_method': self.input_method
            }

            # Add analysis to user's collection
            analysis_ref = db.collection('users').document(user_id).collection('analyses')
            analysis_ref.add(analysis_data)

            # Update user's total analyses count
            user_ref = db.collection('users').document(user_id)
            user_ref.update({'total_analyses': firestore.Increment(1)})

            st.success("✅ Analysis saved to your history!")

        except Exception as e:
            st.error(f"Error saving analysis: {e}")

    def show_history_page(self):
        """Show user's analysis history"""
        st.header("📊 My Analysis History")

        if not db or not st.session_state.user:
            st.error("Database not available")
            return

        user_id = st.session_state.user['localId']

        try:
            # Get user's analyses
            analyses = db.collection('users').document(user_id).collection('analyses').order_by('timestamp',
                                                                                                direction=firestore.Query.DESCENDING).limit(
                20).stream()

            analyses_list = []
            for doc in analyses:
                data = doc.to_dict()
                analyses_list.append(data)

            if analyses_list:
                # Group by reference file
                ref_files = {}
                for analysis in analyses_list:
                    ref_name = analysis.get('reference_file_name', 'Unknown')
                    if ref_name not in ref_files:
                        ref_files[ref_name] = []
                    ref_files[ref_name].append(analysis)

                # Display analyses grouped by reference file
                for ref_file, analyses in ref_files.items():
                    st.subheader(f"🎵 {ref_file}")

                    for i, analysis in enumerate(analyses):
                        timestamp = analysis['timestamp'].strftime('%Y-%m-%d %H:%M') if analysis.get(
                            'timestamp') else 'Unknown'

                        with st.expander(f"Analysis #{len(analyses) - i} - {timestamp}"):
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                pitch_val = analysis.get('pitch_deviation')
                                if pitch_val is not None:
                                    st.metric("Pitch Deviation", f"{pitch_val:.1f} cents")
                                else:
                                    st.metric("Pitch Deviation", "N/A")

                            with col2:
                                rms_val = analysis.get('rms_deviation')
                                if rms_val is not None:
                                    st.metric("Volume Consistency", f"{rms_val:.3f}")
                                else:
                                    st.metric("Volume Consistency", "N/A")

                            with col3:
                                spectral_val = analysis.get('spectral_centroid_deviation')
                                if spectral_val is not None:
                                    st.metric("Timbre Match", f"{spectral_val:.1f}")
                                else:
                                    st.metric("Timbre Match", "N/A")

                            st.info(f"Input Method: {analysis.get('input_method', 'Unknown')}")
            else:
                st.info("📝 No analysis history found. Start analyzing some audio to see your progress!")

        except Exception as e:
            st.error(f"Error loading history: {e}")

    def show_profile_page(self):
        """Show user profile"""
        st.header("👤 User Profile")

        if not db or not st.session_state.user:
            st.error("Database not available")
            return

        user_id = st.session_state.user['localId']

        try:
            user_doc = db.collection('users').document(user_id).get()
            if user_doc.exists:
                user_data = user_doc.to_dict()

                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**👤 Name:** {user_data.get('name', 'N/A')}")
                    st.info(f"**📧 Email:** {user_data.get('email', 'N/A')}")

                with col2:
                    created_at = user_data.get('created_at')
                    if created_at:
                        st.info(f"**📅 Member Since:** {created_at.strftime('%Y-%m-%d')}")
                    else:
                        st.info("**📅 Member Since:** Unknown")
                    st.info(f"**📊 Total Analyses:** {user_data.get('total_analyses', 0)}")

        except Exception as e:
            st.error(f"Error loading profile: {e}")

    def logout(self):
        """Handle user logout"""
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

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


if __name__ == "__main__":
    main()