import streamlit as st
st.set_page_config(page_title="Melody Mentor", layout='wide')

import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import os
from datetime import datetime
from audio_analysis import load_audio, extract_features, compare_features, give_feedback

import firebase_admin
from firebase_admin import credentials, firestore, auth
import pyrebase
import json

from main import AuthHandler

firebaseConfig = {
    "apiKey": "AIzaSyB8roQ3y_BI-FCHySoTgmGYo8CJo_RZP3c",
    "authDomain": "melody-mentor-2dac5.firebaseapp.com",
    "databaseURL": None,
    "projectId": "melody-mentor-2dac5",
    "storageBucket": "melody-mentor-2dac5.firebasestorage.app",
    "messagingSenderId": "255001609512",
    "appId": "1:255001609512:web:f2e748e5da955037323c1d"
}


@st.cache_resource
def init_firebase():
    try:
        firebase = pyrebase.initialize_app(firebaseConfig)
        auth_client = firebase.auth()

        # initialize firebase admin
        if not firebase_admin._apps:
            cred = credentials.Certificate("serviceAccountKey.json")
            firebase_admin.initialize_app(cred)

        db = firestore.client()
        return auth_client, db

    except Exception as e:
        st.error(f"Firebase initialization error: {e}")
        return None, None


auth_client, db = init_firebase()


class AuthHandle:
    def __init__(self):
        self.auth = auth_client

    def sign_up(self, email, password, name):
        # try:
        user = self.auth.create_user_with_email_and_password(email, password)

        # create user profile in firestore
        if db:
            user_ref = db.collection('users').document(user['localId']).document(user['localId'])
            user_ref.set({
                'name': name,
                'email': email,
                'created_at': datetime.now(),
                'total_analyses': 0
            })
        return user, None

    # except Exception as e:
    #     return None, str(e)

    def sign_in(self, email, password):
        user = self.auth.sign_in_with_email_and_password(email, password)
        return user, None


class Frontend:
    def __init__(self):

        if "logged_in" not in st.session_state:
            st.session_state.logged_in = False
        if 'user' not in st.session_state:
            st.session_state.user = None

        # initialize auth handler
        if auth_client:
            self.auth_handler = AuthHandler(auth_client)
        else:
            self.auth_handler = None
            st.error("Firebase not initialized properly")

        # check authorization status
        if not st.session_state.logged_in:
            self.entry_page()
        else:
            self.show_main_app()

    def entry_page(self):
        st.title("🎵 Melody Mentor")
        st.success(
            "Welcome to the application. Here people from any age group can learn how to sing or improve their singing skills")

        # create tabs for login and signup
        tab1, tab2 = st.tabs(['Login', 'Sign Up'])

        with tab1:
            # login
            st.subheader("Login to Your Account")

            with st.form("Login", clear_on_submit=True):
                email = st.text_input("📧 Email")
                password = st.text_input("🔒 Password", type='password')

                col1, col2, col3 = st.columns([2, 1, 1])
                with col2:
                    login_button = st.form_submit_button("Login", use_container_width=False)

                if login_button and self.auth_handler:
                    if email and password:
                        user, error = self.auth_handler.sign_in(email, password)

                        if user:
                            st.session_state.user = user
                            st.session_state.logged_in = True
                            st.success("Login Successful!")
                            st.balloons()
                            st.rerun()

                        else:
                            st.error(f"Login Failed: {error}")

                    else:
                        st.error("Some details are incomplete")

        with tab2:
            # signup
            st.subheader("Create New Account")

            with st.form("Sig_up", clear_on_submit=True):
                name = st.text_input("👤 Full Name")
                email = st.text_input("📧 Email")
                password = st.text_input("🔒 Password", type='password')
                confirm_password = st.text_input("🔒 Confirm Password", type="password")

                col1, col2, col3 = st.columns([2, 1, 1])
                with col2:
                    signup_button = st.form_submit_button("Sign Up", use_container_width=False)

                if signup_button and self.auth_handler:
                    if name and email and password and confirm_password:
                        if password == confirm_password:
                            if len(password) >= 6:
                                user, error = self.auth_handler.sign_up(email, password, name)
                                if user:
                                    st.success("Account created successfully, Please login.")
                                else:
                                    st.error(f"Sign up failed: {error}")
                            else:
                                st.error("Password must be at least 6 characters")
                        else:
                            st.error("Passwords don't match")
                    else:
                        st.error("Please fill in all fields")

    def show_main_app(self):
        # main application after authentication

        col1, col2, col3 = st.columns([3,1,1])

        with col1:
            st.title("🎵 Melody Mentor")

        with col2:
            user_name = st.session_state.user.get('name','user')
            st.write(f"Welcome {user_name}!")

        with col3:
            if st.button("Logout", use_container_width=True)
                self.logout()

        with st.sidebar:
            st.header("Navigation")
            page = st.selectbox(
                "Choose a page:",
                ["🎤 Analyze Singing", "📊 My History", "👤 Profile"],
                key = 'navigation'
            )

            # page routing
            if page == "🎤 Analyze Singing":
                self.show_analysis()
            elif page == "📊 My History":
                self.show_history()
            elif page == "👤 Profile":
                self.show_profile()

    def refFile(self):
        self.ref_audio_file = st.file_uploader(
            label= "Upload your reference audio file",
            type = ['wav',"mp3"],
            accept_multiple_files= False,
            key = "ref_file_uploader"
        )

    def inputMethod(self):
        st.subheader("Your singing sample")
        self.input_method = st.radio(
            "Choose how to provide your singing sample:",
            ["Record Audio", "Upload Audio File"],
            key = "input_method"
        )

        if self.input_method == "Record Audio":
            self.user_audio_file = st.audio_input("Record your singing sample", key = "user_audio_input")

        else:
            self.user_uploaded_file = st.file_uploader(
                label = "Upload your singing sample",
                type = ['wav',"mp3"],
                accept_multiple_files = False,
                key = "user_uploaded_file"
            )


    def logout(self):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


if __name__ == "__main__":
    Frontend()