# Multimodal Emotion Recognition from Speech and Text  
### **by Aditya Gupta**

##  Motivation

Emotion plays a central role in how humans communicate. Beyond the words we use, our tone, pace, and inflection all carry emotional weight. Yet, most machines today only "understand" text literally — they lack awareness of the emotion behind it.

I picked this topic to explore whether combining both **speech** and **text** can help a model understand the *intent* behind what someone is saying. This felt like an exciting problem because of the wide scope of problems it can solve and the way it can bridge the emotional gap between a human and a machine

Some of the real world applications may inlcude:
- **Voice Assistants** : AI Assisstants like Alexa can be better able to learn users behaviour and respond according to the mood
- **Mental health monitoring tools** it could be helpful in mental health treatment where audio and textual behaviour can be used to predict human's response.

##  Historical Context: Where Does This Fit in Multimodal Learning?

Multimodal learning — the idea of combining different data types like audio, text, and vision — has gained major traction in the past decade. Earlier systems treated tasks like sentiment analysis or emotion detection using either speech or text, but not both.

In audio-only emotion recognition, traditional models focused on prosodic features such as pitch, energy, and Mel-Frequency Cepstral Coefficients (MFCCs). For example, the work by **Lee and Narayanan (2005)** showed how vocal features could be leveraged to recognize emotions like anger or sadness in speech.

On the text side, emotion detection models initially relied on sparse features like bag-of-words and TF-IDF. However, this approach ignored word semantics and contextual dependencies. Later work such as **Felbo et al. (2017)** introduced **DeepMoji**, which used emoji-labeled tweets to pre-train a model for emotional representation in text.

More recent multimodal approaches include **Poria et al. (2017)** – "A Multi-Level Multimodal Fusion Framework for Emotion Recognition" which shows a clear shift toward integrating multiple modalities to improve contextual understanding. Our work builds on these foundations — using GloVe for semantics, audio features for vocal tone, and early fusion to combine them for a more complete emotional profile.
Hence our approach also aligns with the same principle emotion is complex and best understood when modalities are combined.

##  What I Learned From This Work

This project turned out to be a hands-on crash course in multimodal machine learning. We started with simple pipelines, but ended up learning a wide range of ideas — both conceptual and practical.

We learned how to **balance a highly skewed dataset**, where the "Neutral" class dominated the samples. Designing a fair subset while keeping the emotion distribution realistic taught us how even preprocessing can make or break a model's performance.

On the audio side, we explored low-level signal features like **MFCCs**, **Zero Crossing Rate**, **Spectral Centroid**, and **RMS Energy** — and understood not just how to extract them, but what they mean and why they matter emotionally. Building a 16-dimensional audio feature vector helped us capture emotion-related variations in tone and loudness.

We also revisited and applied **classical classification algorithms** — Logistic Regression, KNN, and Random Forest — understanding when they work well and where they fall short, especially in high-overlap, low-data settings.

From the text side, we used **TF-IDF** as a familiar baseline, and moved on to **GloVe embeddings** to bring in semantic information. We learned how simple averaging of word vectors can still give reasonably meaningful sentence representations.

Most importantly, we understood the importance of **fusion techniques** — especially **early fusion** — and how combining different modalities can overcome the limitations of unimodal systems. Even without deep learning, we saw real benefits in integrating speech and text.

This project helped connect ideas across audio processing, NLP, data balancing, and model evaluation — and gave us a better appreciation of what it takes to make machine learning models understand human emotions.

## Reflections – Surprises & Scope for Improvement

### What Surprised me
1. **Emotion confusion and immense overlapping of emotions.** We assumed emotions like "joy" or "anger" would be easy to classify, but even they often got confused with "neutral" — especially in short or flatly delivered utterances.
2. **Surprising overlap of emotions like joy and anger during classification due to tonal characterstics**
3. **"Neutral" isn't actually neutral.**  Many utterances labeled as neutral were semantically rich or emotionally ambiguous. The dominance of this class skewed early models and made evaluation tricky.

### Scope for Improvement

- **Contextual embeddings** like BERT or RoBERTa could capture sentence structure, sarcasm, and negation better than GloVe.
- **Visual modality** (e.g., facial expressions or gestures from video) could provide additional cues, especially for ambiguous emotions like "surprise" or "disgust".
- **Late fusion** or **attention-based multimodal architectures** could offer more flexible and powerful decision-making than simple early fusion.

##  References

Below are some research works that align with parts of our pipeline — particularly in using audio, text, or their fusion for emotion recognition. These helped contextualize and validate our choices:

- **Lee & Narayanan (2005)** – *Toward detecting emotions in spoken dialogs*  
  One of the earliest works using prosodic and MFCC-based features for emotion recognition in speech.  
  [https://ieeexplore.ieee.org/document/1521423](https://ieeexplore.ieee.org/document/1521423)

- **Zhang et al. (2017)** – *Learning Affect from Speech: A Survey*  
  A comprehensive review of audio-based emotion recognition techniques, including classical classifiers with MFCCs.  
  [https://arxiv.org/abs/1701.02510](https://arxiv.org/abs/1701.02510)

- **Akçay & Oğuz (2020)** – *Speech Emotion Recognition: Emotional Models, Databases, Features, Preprocessing Methods, Supporting Modalities, and Classifiers*  
  Reviews common approaches in speech emotion recognition using features like MFCC, ZCR, and classical ML models like SVM and RF.  
  [https://www.sciencedirect.com/science/article/pii/S0893608020301952](https://www.sciencedirect.com/science/article/pii/S0893608020301952)

- **Zhao et al. (2019)** – *Multi-modal Emotion Recognition for One-minute Gradual Emotion Challenge*  
  Uses early fusion of text and audio features (including MFCC and GloVe embeddings) to classify emotions.  
  [https://arxiv.org/abs/1910.05756](https://arxiv.org/abs/1910.05756)

- **Albornoz et al. (2017)** – *Unified Approach for Emotion Recognition in Twitter Using Deep Learning*  
  While focused on text only, it motivates the use of GloVe-style embeddings in emotion classification.  
  [https://arxiv.org/abs/1708.03902](https://arxiv.org/abs/1708.03902)

We also referenced:
- GloVe Embeddings: [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)  
- `librosa` for audio feature extraction  
- `scikit-learn` for classification and evaluation tools

