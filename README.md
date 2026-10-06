# Sign Language Detector

Real-time hand sign recognition in Python. A webcam feed goes through MediaPipe Hands (21 landmarks), a custom normalisation step, and a k-NN classifier, with a 10-frame majority vote to stop flicker.

*Originally a Class 12 group project in Python + NumPy; rebuilt independently so I understand every stage.*

![demo](demo.gif)

## What it recognises

8 classes: `one`, `two` (also peace/victory), `three`, `four`, `five`, `thumbs_up`, `thumbs_down`, and `none` (fists, relaxed or random hand poses, so the model doesn't label every hand as a sign). Works with either hand, palm or back of the hand facing the camera.

## Pipeline

1. **OpenCV** reads the webcam and mirrors the frame.
2. **MediaPipe Hands** returns 21 landmarks per hand.
3. **Normalisation** (`features.py`): translate so the wrist is the origin, scale by the wrist-to-middle-finger-base distance, **no rotation**. Rotating to a fixed angle would erase the difference between thumbs up and thumbs down.
4. **k-NN (k=5)** classifies the 42 features. Random Forest was compared.
5. **Smoothing:** majority vote over the last 10 frames.

## Results

Trained on three sessions of my own hands (s1 right hand, s2 left hand, s3 both hands in different light and distance). Tested on f1: a friend's hands, both hands, both views, never used for training.

| | k-NN (k=5) | Random Forest |
|---|---|---|
| Leave-one-session-out (my hands only) | 98.5% | 98.4% |
| **New person (f1)** | **91.5%** | 86.7% |

New-person k-NN, per class (F1 score): thumbs_up 0.997, thumbs_down 0.984, one 0.966, two 0.961, none 0.918, five 0.860, three 0.818, four 0.777.

Full output: [`results/train_output.txt`](results/train_output.txt)

**How to read these numbers**
- The 98.5% is **not** a new-person result. All three sessions are the same person. The honest headline is 91.5% on one unseen person.
- f1 is one person, so the error bar is wide.
- I picked k-NN after seeing the f1 results, which is a small leak. Both models are reported