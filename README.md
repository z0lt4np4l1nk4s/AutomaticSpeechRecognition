# Automatic Speech Recognition on the Free Spoken Digit Dataset (FSDD)

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Dataset](#dataset)
3. [ASR Pipeline](#asr-pipeline)
4. [ASR History](#asr-history)
5. [Algorithms Implemented](#algorithms-implemented)
6. [Quick Start](#quick-start)
7. [Project Structure](#project-structure)
8. [Experimental Results](#experimental-results)
9. [Record‑Your‑Own Demo](#record‑your‑own-demo)
10. [Limitations and Future Work](#limitations-and-future-work)

---

## 1. Problem Statement

The goal is to **recognise isolated spoken digits (0–9)** from short audio clips.  We build and compare three classical ASR approaches:

| Abbrev. | Algorithm                                       | Key idea                                                                                               |
| ------- | ----------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **DTW** | 1‑Nearest‑Neighbour with *Dynamic Time Warping* | Aligns test MFCC sequence to each training sequence and picks the smallest distance                    |
| **HMM** | *Gaussian‑Mixture* Hidden Markov Models         | Learns state‑transition and emission probabilities per digit, then scores a test sequence via Viterbi   |
| **CNN** | 2‑layer *Convolutional Neural Network*          | Classifies fixed‑size log‑Mel spectrogram “images”.                                                   |

---

## 2. Dataset – Free Spoken Digit Dataset (FSDD)

- **Size:** \~3,000 WAV files, 8 kHz, mono, ≈1 s per clip
- **Speakers:** 6
- **Licence:** Creative Commons BY 4.0
- **Repo:** [https://github.com/Jakobovski/free-spoken-digit-dataset](https://github.com/Jakobovski/free-spoken-digit-dataset)

---

## 3. ASR Pipeline

1. **Pre‑processing:**  Normalise volume, optional Voice Activity Detection (VAD)
2. **Feature Extraction:**
   - *MFCC + ΔMFCC* for DTW/HMM
   - *Log‑Mel Spectrogram* (64×128) for the CNN
3. **Acoustic Model & Classifier:** DTW, GMM‑HMM, or CNN
4. **Decoding / Inference:**  1‑NN vote (DTW), Viterbi (HMM), or Softmax (CNN)
5. **Evaluation:**  Digit Accuracy, Confusion Matrix

---

## 4. ASR History
- **1952 – Audrey:** Bell Labs’ first digit recogniser
- **1960s – DTW:** dynamic time warping aligns variable speaking speed
- **1980s – GMM‑HMM:** Rabiner & colleagues formalise the HMM‑based pipeline
- **2012 – Deep Speech / CTC:** deep neural networks replace GMMs; CTC enables end‑to‑end training
- **2020 – Self‑supervised pre‑training:** Wav2Vec 2.0, HuBERT, Whisper learn from thousands of hours of unlabeled speech

---

## 5. Algorithms Implemented

### 5.1 Dynamic Time Warping (DTW)

- *Pros*: Zero training, interpretable distance path.
- *Cons*: O(N·T²) runtime, sensitive to noise.

### 5.2 Gaussian‑Mixture HMM

- 5 hidden states, 4 Gaussians/state, diagonal covariance.
- Trained per digit with Baum‑Welch, inference via Viterbi log‑likelihood

### 5.3 Convolutional Neural Network

- 1 × 64 × 128 log‑Mel input → Conv32 → Conv64 → GAP → Linear10.
- \~40 k parameters; trained 8 epochs with Adam (lr = 1e‑3).

---

## 6. Quick Start

Clone the repository
```bash
$ git clone https://github.com/z0lt4np4l1nk4s/AutomaticSpeechRecognition
$ cd AutomaticSpeechRecognition
```
Environment setup
*For Linux/macOS:*
```bash
# Navigate to your project directory
# Create a Python virtual environment
python3 -m venv venv

# Activate the environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

*For Windows:*
```bash
# Navigate to your project directory in PowerShell or Command Prompt
# Create a Python virtual environment
python -m venv venv

# Activate the environment
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Launch Jupyter / VS Code and open **notebook.ipynb** them eun the notebook **top‑to‑bottom**

---

## 7. Project Structure

```text
├── notebook.ipynb
├── requirements.txt
└── README.md
```

---

## 8. Experimental Results

| Model    | Accuracy (FSDD test split) |
| -------- | -------------------------- |
| DTW 1‑NN | **60 %**                   |
| GMM‑HMM  | **99 %**                   |
| CNN      | **92 %**                   |

Confusion matrices are generated automatically in the notebook (`sklearn.metrics.ConfusionMatrixDisplay`).

---

## 9. Record‑Your‑Own Demo

```python
# Inside the notebook, Section 7:
import sounddevice as sd, scipy.io.wavfile as wav
fs = 8000; seconds = 1
sd.rec(int(seconds*fs), samplerate=fs, channels=1)
# speak your digit → press ▶
```

Predictions from all three models are printed; compare which algorithm copes best with an **unseen speaker**

---

## 10. Limitations & Future Work

- **Isolated digits** only – no continuous speech or language model
- Small dataset → models may overfit speakers & noise profile
- HMM configuration (states, mixtures) was not exhaustively tuned
- CNN uses fixed‑length input; RNN/CTC would allow variable‑length sequences

**Next steps**: data augmentation, fine‑tuning `wav2vec2‑base` on FSDD, adding CTC loss, streaming inference.

---

