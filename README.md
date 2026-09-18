# 🎵 Melody Mentor

### AI-Assisted Singing Analysis & Vocal Improvement Platform

**Melody Mentor** is an AI-assisted singing analysis application designed to help singers understand and improve their vocal performance by comparing their singing with a reference recording.

Instead of simply telling a user whether their singing sounds good or bad, Melody Mentor analyzes measurable audio characteristics such as **pitch, volume consistency, and timbre** and converts those measurements into understandable feedback.

> 🎤 **Sing smarter. Practice better. Improve with data.**

---

## ✨ Overview

Learning to sing traditionally relies heavily on subjective feedback from teachers, peers, or the singer themselves. This can make it difficult for beginners to identify exactly where they are going wrong.

Melody Mentor approaches this problem through **audio signal processing and computational analysis**.

The application takes:

* 🎵 A **reference audio recording**
* 🎤 The user's **singing recording**

and performs an audio comparison to identify differences in important vocal characteristics.

The current system analyzes:

* **Pitch / frequency**
* **Volume consistency**
* **Timbre / spectral characteristics**

The results are presented through visualizations, numerical metrics, and personalized feedback.

---

## 🎯 Problem Statement

Many beginner singers struggle to identify *why* their singing sounds different from a reference performance.

For example:

* Am I singing the correct notes?
* Am I going flat or sharp?
* Is my volume consistent?
* Does the overall vocal character differ significantly from the reference?
* Am I actually improving with practice?

Existing music applications often focus on recording, playback, or entertainment rather than providing an accessible analytical layer for singers.

**Melody Mentor aims to bridge this gap by turning a singing recording into measurable information and actionable feedback.**

---

## 💡 Key Features

### 🎤 1. Reference-Based Singing Analysis

Users can upload a reference `.wav` or `.mp3` file representing the song or vocal performance they want to learn.

The reference becomes the baseline against which the user's singing is analyzed.

---

### 🎙️ 2. Record or Upload Your Singing

Users can provide their singing in two ways:

* Record directly through the application
* Upload an existing audio file

This makes the system useful both for immediate practice and for analyzing previously recorded performances.

---

### 🎼 3. Pitch Analysis

Melody Mentor extracts the fundamental frequency (**F0**) from the audio using **Librosa's PYIN algorithm**.

The extracted pitch information allows the application to compare the user's notes with those present in the reference recording.

Pitch deviation is represented in **cents**, allowing relatively small differences in pitch to be quantified.

---

### 🔊 4. Volume / Energy Analysis

The system calculates **RMS energy** from the audio signal to analyze vocal volume and consistency.

This helps identify differences between the reference and user's vocal energy.

---

### 🎨 5. Timbre Analysis

Melody Mentor uses the **spectral centroid** as an indicator of the spectral characteristics or perceived brightness of the audio.

This provides another dimension for comparing the user's vocal recording with the reference.

---

### 📊 6. Visual Audio Comparison

The application generates visual comparisons between the reference recording and the user's singing, including:

* Audio waveforms
* Pitch contours
* RMS energy / volume

This allows users to visually understand differences that may otherwise be difficult to hear.

---

### 🎯 7. Personalized Feedback

After extracting and comparing the audio features, Melody Mentor generates feedback based on the detected differences.

The goal is to translate technical audio measurements into feedback that is understandable and useful for a singer.

---

### 🔐 8. User Authentication

The application integrates **Firebase Authentication** for user registration and login.

User information is maintained through Firebase/Firestore, allowing each user's analysis data to be associated with their account.

---

### 📚 9. Analysis History

Melody Mentor stores previous analyses in **Cloud Firestore**.

Users can revisit their previous singing analyses and view metrics such as:

* Pitch deviation
* Volume consistency
* Timbre match
* Input method
* Analysis timestamp

This creates the foundation for tracking improvement over time.

---

## 🧠 How It Works

The overall pipeline can be summarized as:

```text
          ┌─────────────────────┐
          │   Reference Audio   │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │   Audio Loading     │
          │      Librosa        │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ Feature Extraction  │
          │                     │
          │ • Pitch (PYIN)      │
          │ • RMS Energy        │
          │ • Spectral Centroid │
          └──────────┬──────────┘
                     │
                     │
          ┌──────────▼──────────┐
          │ Compare Features    │
          │                     │
          │ Reference ↔ User    │
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │ Feedback Generation │
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │ Visualization +     │
          │ Technical Metrics   │
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │ Save to Firestore   │
          └─────────────────────┘
```

---

## 🛠️ Technology Stack

| Technology                  | Purpose                                     |
| --------------------------- | ------------------------------------------- |
| **Python**                  | Core application and audio-processing logic |
| **Streamlit**               | Interactive web application interface       |
| **Librosa**                 | Audio loading and feature extraction        |
| **NumPy**                   | Numerical processing                        |
| **Matplotlib**              | Audio and feature visualizations            |
| **Firebase Authentication** | User authentication                         |
| **Firebase Admin SDK**      | Backend Firebase integration                |
| **Cloud Firestore**         | User profiles and analysis history          |
| **Requests**                | Communication with Firebase REST APIs       |

The repository's current dependency list includes Streamlit, Firebase Admin, Requests, Librosa, Matplotlib, and NumPy.

---

## 📂 Project Structure

```text
melody-mentor/
│
├── .streamlit/
│   └── secrets configuration
│
├── song refrence files/
│   └── Reference audio files
│
├── main.py
│   └── Main Streamlit application
│       ├── Authentication
│       ├── Navigation
│       ├── Audio input
│       ├── Singing analysis
│       ├── Visualization
│       ├── Feedback
│       └── Analysis history
│
├── audio_analysis.py
│   └── Core audio-analysis functions
│       ├── Audio loading
│       ├── Pitch extraction
│       ├── RMS calculation
│       ├── Spectral analysis
│       ├── Feature comparison
│       └── Feedback generation
│
├── singer.py
│   └── Streamlit/frontend experimentation and singing-analysis interface
│
├── prctc.py
│   └── Experimental/practice implementation
│
├── requirements.txt
│   └── Python dependencies
│
└── README.md
```

---

# 🔬 Audio Analysis Pipeline

## 1. Audio Loading

Audio files are loaded using Librosa.

The system obtains:

```text
Audio waveform + Sampling Rate
```

If the reference and user recordings have different sampling rates, the user's audio is resampled to match the reference before comparison.

---

## 2. Pitch Extraction

Melody Mentor uses **PYIN** to estimate the fundamental frequency of the singing voice.

The current pitch range is configured between approximately:

```text
C2 → C7
```

This provides the system with a pitch contour representing how the singer's vocal frequency changes throughout the recording.

---

## 3. RMS Energy

RMS energy is extracted from the waveform to represent the signal's energy over time.

It is used as a proxy for **volume/energy consistency**.

---

## 4. Spectral Centroid

The spectral centroid is calculated to capture information about the distribution of frequencies within the audio.

It is used as one of the indicators for comparing the **timbre** of the reference and user's recording.

---

## 5. Feature Comparison

The extracted features from both recordings are compared:

```text
Reference Features
        │
        ├── Pitch
        ├── RMS Energy
        └── Spectral Centroid
                    │
                    ▼
             COMPARISON
                    ▲
                    │
        ├── Pitch
        ├── RMS Energy
        └── Spectral Centroid
        │
User Features
```

The application then calculates comparison metrics and generates feedback.

---

# 📈 Metrics

The application currently exposes metrics including:

### 🎵 Pitch Deviation

Measured in **cents**.

A lower deviation indicates that the user's estimated pitch is closer to the reference pitch.

### 🔊 Volume Consistency

Based on differences in RMS energy between the recordings.

### 🎨 Timbre Match

Based on differences in spectral characteristics using spectral centroid information.

These metrics are displayed directly within the Streamlit interface.

---

# 🖥️ Application Flow

```text
1. Create an account / Log in
              ↓
2. Select "Analyze Singing"
              ↓
3. Upload reference song
              ↓
4. Record or upload your singing
              ↓
5. Click "Analyze My Singing"
              ↓
6. Audio preprocessing
              ↓
7. Feature extraction
              ↓
8. Feature comparison
              ↓
9. Generate feedback
              ↓
10. Display graphs + metrics
              ↓
11. Save analysis to history
```

---

# 🚀 Installation & Setup

## Prerequisites

Make sure you have:

* Python 3.x
* pip
* A Firebase project
* Firebase Authentication enabled
* Cloud Firestore configured

---

## 1. Clone the Repository

```bash
git clone https://github.com/IshitaPhogat/melody-mentor.git
cd melody-mentor
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure Firebase

Melody Mentor uses Firebase Authentication and Cloud Firestore.

Firebase credentials should be provided through Streamlit secrets rather than hard-coded service-account credentials.

Create:

```text
.streamlit/
└── secrets.toml
```

and configure the required Firebase values according to your Firebase project.

> ⚠️ **Security Note:** Never commit Firebase private keys, service-account credentials, passwords, or other secrets to GitHub.

---

## 5. Run the Application

```bash
streamlit run main.py
```

The application will start locally and provide a URL through which the Streamlit interface can be accessed.

---

# 🔮 Future Scope

Melody Mentor is currently focused primarily on **singing-performance analysis**, but the project has significant potential for expansion.

During research for this project, I explored existing work in **audio enhancement and music source separation**. There are already several projects that demonstrate how deep-learning-based systems can perform tasks such as:

* Background-noise reduction
* Speech enhancement
* Vocal isolation
* Background-music removal
* Music source separation
* Instrument separation
* Stem extraction

For example, existing open-source research and implementations demonstrate separation of vocals and accompaniment, while other systems can separate stems such as **vocals, drums, bass, and other instruments** using neural networks.

### 🎧 Planned Audio Enhancement

A major future direction for Melody Mentor is to incorporate these ideas directly into the application.

The planned pipeline could become:

```text
             User Audio
                  │
                  ▼
       ┌─────────────────────┐
       │ Audio Preprocessing │
       └──────────┬──────────┘
                  │
                  ▼
       ┌─────────────────────┐
       │ Noise Reduction     │
       │ & Voice Enhancement │
       └──────────┬──────────┘
                  │
                  ▼
       ┌─────────────────────┐
       │ Vocal / Instrument  │
       │ Separation          │
       └──────────┬──────────┘
                  │
                  ▼
       ┌─────────────────────┐
       │ Singing Analysis    │
       │                     │
       │ Pitch               │
       │ Volume              │
       │ Timbre              │
       └──────────┬──────────┘
                  │
                  ▼
       ┌─────────────────────┐
       │ Personalized        │
       │ Singing Feedback    │
       └─────────────────────┘
```

This would allow Melody Mentor to work with much cleaner vocal signals, particularly when a singer records while a song's instrumental track is playing in the background.

Future versions could investigate models and approaches used in modern **music source separation**, including architectures such as **U-Net-based models, MDX-Net, Demucs, and related source-separation approaches**, depending on computational requirements and licensing considerations.

### 🎶 Possible Future Features

* 🎧 Automatic background-noise removal
* 🎤 Vocal isolation from instrumental tracks
* 🥁 Instrument/stem separation
* 🎹 Separate vocals, bass, drums and other instruments
* 🎼 Automatic detection of song sections
* 🎵 Note-by-note pitch feedback
* ⏱️ Rhythm and timing analysis
* 📈 Progress tracking across multiple sessions
* 🤖 More advanced AI-generated vocal coaching
* 📱 Mobile application
* ⚡ Real-time singing feedback

---

# 🧪 Current Limitations

Melody Mentor is an evolving project and there are several areas that can be improved.

### Audio Quality

Pitch and timbre analysis can be affected by:

* Background noise
* Instrumental music
* Echo/reverberation
* Poor microphones
* Multiple voices
* Different recording environments

### Alignment

The reference and user's performance may not always be perfectly synchronized, which can influence direct feature comparisons.

### Feature Coverage

The current system primarily focuses on:

```text
Pitch
Volume
Timbre
```

More advanced musical characteristics such as rhythm, pronunciation, vibrato, timing, and detailed note-level accuracy can be incorporated in future versions.

### Source Separation

The current implementation does not yet perform full neural audio source separation. Integrating such models is part of the planned future development rather than a current feature.

---

# 📚 Research & Inspiration

Melody Mentor was developed with an interest in combining **music, signal processing, and AI-assisted learning**.

Research into related audio-processing projects showed that the broader field already includes systems capable of:

* Noise reduction
* Speech separation
* Background music removal
* Vocal isolation
* Instrument separation
* Multi-stem music demixing

For example:

* **Music Source Separation** research implementations demonstrate separation of vocals and accompaniment using neural networks.
* **Looking to Listen** explores deep neural networks for noise reduction, background-music removal, and speech separation.
* Other open-source implementations use models such as **MDX-Net and Demucs** to separate vocals, drums, bass, and other musical components.

These projects helped identify a natural future direction for Melody Mentor: **combining clean-vocal extraction with singing-performance analysis.**

---

# 🎓 Project Objective

The long-term vision of Melody Mentor is to create an accessible **AI-powered personal vocal coach**.

Rather than replacing a human singing teacher, the goal is to provide singers with an additional tool that can offer:

> **Objective measurements → understandable feedback → repeated practice → measurable improvement**

---

# 🤝 Contributing

Contributions, ideas, and suggestions are welcome.

If you would like to contribute:

```bash
# Fork the repository

# Create a new branch
git checkout -b feature/your-feature

# Make your changes

# Commit
git commit -m "Add your feature"

# Push
git push origin feature/your-feature
```

Then open a Pull Request.

---

# 👩‍💻 Author

**Ishita Phogat**

B.Tech Computer Science Engineering
Specialization: Data Science & Artificial Intelligence

---

# ⭐ Acknowledgements

This project makes use of several open-source technologies, particularly:

* [Streamlit](https://streamlit.io/)
* [Librosa](https://librosa.org/)
* [NumPy](https://numpy.org/)
* [Matplotlib](https://matplotlib.org/)
* [Firebase](https://firebase.google.com/)

---

## 📌 Project Status

🟢 **Active Development**

Melody Mentor currently provides reference-based singing analysis with audio feature extraction, comparison, visualization, feedback, authentication, and analysis history.

🚧 **Next major direction:**
Integrating **audio enhancement and source-separation techniques** to isolate cleaner vocals and eventually enable deeper analysis of vocals, background music, and individual instruments.

---

<p align="center">

### 🎵 Melody Mentor

**Your voice. Your data. Your progress.**

</p>
