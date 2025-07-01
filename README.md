# Automatic Speech Recognition on the Free Spoken Digit Dataset (FSDD)

> **Note / Napomena:** The notebook code and this README are written in English to keep technical terminology precise, while the accompanying video walkthrough is recorded in **Croatian** (HR) as required by the course instructions.

---

## Table of Contents

1. [Problem Statement](#1)
2. [Dataset](#2)
3. [Background: The ASR Pipeline](#3)
4. [Algorithms Implemented](#4)
5. [Quick Start](#5)
6. [Project Structure](#6)
7. [Experimental Results](#7)
8. [Record‑Your‑Own Demo](#8)
9. [Environment & Dependencies](#9)
10. [Limitations & Future Work](#10)
11. [References](#11)

---



## 1. Problem Statement

The goal is to **recognise isolated spoken digits (0–9)** from short audio clips.  We build and compare three classical ASR approaches:

| Abbrev. | Algorithm                                       | Key idea                                                                                               |
| ------- | ----------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **DTW** | 1‑Nearest‑Neighbour with *Dynamic Time Warping* | Aligns test MFCC sequence to each training sequence and picks the smallest distance.                   |
| **HMM** | *Gaussian‑Mixture* Hidden Markov Models         | Learns state‑transition and emission probabilities per digit, then scores a test sequence via Viterbi. |
| **CNN** | 2‑layer *Convolutional Neural Network*          | Classifies fixed‑size log‑Mel spectrogram “images”.                                                    |

We purposely avoid end‑to‑end Transformer/Whisper models to focus on **foundational ASR concepts**.

---



## 2. Dataset – Free Spoken Digit Dataset (FSDD)

- **Size:** \~2,000 WAV files, 8 kHz, mono, ≈1 s per clip
- **Speakers:** 6
- **Licence:** Creative Commons BY 4.0
- **Repo:** [https://github.com/Jakobovski/free-spoken-digit-dataset](https://github.com/Jakobovski/free-spoken-digit-dataset)

We add **10 personal recordings** (one per digit) to test generalisation.

---



## 3. Background: The ASR Pipeline



1. **Pre‑processing:**  Normalise volume, optional Voice Activity Detection (VAD).
2. **Feature Extraction:**
   - *MFCC + ΔMFCC* for DTW/HMM
   - *Log‑Mel Spectrogram* (64×128) for the CNN
3. **Acoustic Model & Classifier:** DTW, GMM‑HMM, or CNN.
4. **Decoding / Inference:**  1‑NN vote (DTW), Viterbi (HMM), or Softmax (CNN).
5. **Evaluation:**  Digit Accuracy, Confusion Matrix.

---



## 4. Algorithms Implemented

### 4.1 Dynamic Time Warping (DTW)

- *Pros*: Zero training, interpretable distance path.
- *Cons*: O(N·T²) runtime, sensitive to noise.

### 4.2 Gaussian‑Mixture HMM

- 5 hidden states, 4 Gaussians/state, diagonal covariance.
- Trained per digit with Baum‑Welch; inference via Viterbi log‑likelihood.

### 4.3 Convolutional Neural Network

- 1 × 64 × 128 log‑Mel input → Conv32 → Conv64 → GAP → Linear10.
- \~40 k parameters; trained 8 epochs with Adam (lr = 1e‑3).

---



## 5. Quick Start

```bash
# Clone project & go to folder
$ git clone https://github.com/<your-user>/asr-digits.git
$ cd asr-digits

# (Recommended) create a virtual environment
$ python -m venv .venv && source .venv/bin/activate

# Install dependencies
$ pip install -r requirements.txt

# Launch Jupyter / VS Code and open  fsdd_asr_workshop.ipynb
```

Run the notebook **top‑to‑bottom**.  GPU runtime (Colab) speeds up section 5 (CNN).

---



## 6. Project Structure

```text
├── data/                 # FSDD WAV files + personal recordings
├── docs/
│   └── img/              # figures used in README / video
├── fsdd_asr_workshop.ipynb
├── requirements.txt
└── README.md             # this file
```

---



## 7. Experimental Results

| Model    | Accuracy (FSDD test split) |
| -------- | -------------------------- |
| DTW 1‑NN | **88 %**                   |
| GMM‑HMM  | **96 %**                   |
| CNN      | **98 %**                   |

Confusion matrices are generated automatically in the notebook (`sklearn.metrics.ConfusionMatrixDisplay`).

---



## 8. Record‑Your‑Own Demo

```python
# Inside the notebook, Section 7:
import sounddevice as sd, scipy.io.wavfile as wav
fs = 8000; seconds = 1
sd.rec(int(seconds*fs), samplerate=fs, channels=1)
# speak your digit → press ▶
```

Predictions from all three models are printed; compare which algorithm copes best with an **unseen speaker**.

---



## 9. Environment & Dependencies

- Python ≥ 3.9
- `librosa`, `soundfile`, `fastdtw`, `hmmlearn`, `pomegranate`
- `torch`, `torchaudio`, `scikit‑learn`, `sounddevice`, `matplotlib`, `seaborn`

Full list in `requirements.txt`.

---



## 10. Limitations & Future Work

- **Isolated digits** only – no continuous speech or language model.
- Small dataset → models may overfit speakers & noise profile.
- HMM configuration (states, mixtures) was not exhaustively tuned.
- CNN uses fixed‑length input; RNN/CTC would allow variable‑length sequences.

**Next steps**: data augmentation, fine‑tuning `wav2vec2‑base` on FSDD, adding CTC loss, streaming inference.

---



## 11. References

1. Rabiner, L. (1989). *A Tutorial on Hidden Markov Models and Selected Applications.*
2. Müller, M. (2007). *Dynamic Time Warping.*
3. Jakobovski & Koch (2018). *Free Spoken Digit Dataset.*
4. Graves, A. (2012). *Sequence Transduction with RNNs.*
5. O’Shaughnessy, D. (2008). *Invited Paper: Automatic Speech Recognition: History, Methods and Challenges.*

*(Full BibTeX in **`docs/refs.bib`**)*

---

### Credits

Created by **[Your Name]** as part of *Računalno jezikoslovlje* 2025 project at FER.\
Special thanks to FSDD contributors and open‑source library authors.

---

> **How to cite** this repo: `@misc{yourname2025asrDigits, title={ASR on Free Spoken Digit Dataset}, author={Your Name}, year={2025}, howpublished={GitHub}, note={v1.0}}`

