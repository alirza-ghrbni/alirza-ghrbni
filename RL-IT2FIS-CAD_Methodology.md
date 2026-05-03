# Methodology Chapter Draft

## Thesis Title
**A Fuzzy Logic-Based Method for Criminal Activity Detection to Enhance Environmental Security Using Reinforcement Learning**

---

## 1. Overview of the Proposed Method

This thesis proposes **RL-IT2FIS-CAD** (**Reinforcement Learning Optimized Interval Type-2 Fuzzy Inference System for Criminal Activity Detection**), a hybrid deep neuro-fuzzy reinforcement learning architecture for binary violence detection in surveillance videos. The framework is specifically designed for the **RWF-2000** dataset and targets three practical requirements: (i) lightweight computation, (ii) interpretability, and (iii) robustness against overfitting.

The architecture combines three complementary components:

1. **Deep spatio-temporal feature learning** to model visual evidence of violent interactions.
2. **Interval Type-2 fuzzy inference** to perform uncertainty-aware, interpretable decision-making.
3. **Fuzzy Q-Learning optimization** to adapt fuzzy rule weights and decision thresholds.

Importantly, fuzzy logic is the **core decision-making mechanism**, while reinforcement learning is used to **optimize the fuzzy decision process** rather than replace it. This design preserves alignment with the thesis title, where fuzzy logic is primary and reinforcement learning is an optimization layer.

---

## 2. Dataset

The study uses **only the RWF-2000 dataset**. RWF-2000 is a binary violence detection benchmark containing surveillance-style clips labeled as:

- **Violent**
- **Non-violent**

Training and evaluation should follow the official (or most commonly adopted) train-test split protocol to ensure comparability with prior work. No external dataset is mixed with RWF-2000 in either training or testing.

Because RWF-2000 is relatively small for deep video learning, the model is intentionally lightweight and regularized to minimize overfitting while maintaining high discriminative power.

---

## 3. Video Preprocessing

Each input clip is transformed into a fixed-length sequence using a consistent pipeline:

1. **Uniform frame sampling** of **16 or 32 frames** per clip.
2. **Frame resizing** to **224×224** (preferred) or **256×256**.
3. **Normalization** with ImageNet mean and standard deviation.
4. **Light augmentation** during training only:
   - random horizontal flip,
   - random crop,
   - mild color jitter,
   - temporal jitter,
   - random frame dropping.

Augmentation is deliberately moderate because aggressive transformations can distort motion cues that are essential for violence characterization.

---

## 4. ConvNeXt-Tiny + Dual Attention Feature Extractor

### 4.1 Backbone Selection

**ConvNeXt-Tiny** is selected as the spatial encoder due to its favorable trade-off between representational strength and parameter efficiency compared with larger ConvNeXt variants. The backbone is initialized using **ImageNet pretrained weights**.

### 4.2 Fine-Tuning Strategy

To reduce overfitting:

- early and middle ConvNeXt stages are frozen,
- only the final ConvNeXt block is fine-tuned,
- Dual Attention remains trainable,
- temporal modules and classifier layers remain trainable.

### 4.3 Dual Attention Design

The Dual Attention block includes:

- **Channel Attention**: emphasizes feature channels strongly correlated with violent behavior.
- **Spatial Attention**: emphasizes frame regions with physically aggressive interaction (e.g., pushing, hitting, crowd struggle).

For each sampled frame, the module outputs one spatial feature vector. Across the clip, this forms a sequence of frame-wise descriptors.

---

## 5. Feature Compression Layer

A lightweight projection layer is inserted after ConvNeXt-Tiny + Dual Attention:

- input dimension: ConvNeXt output feature size,
- output dimension: **256 or 512**,
- activation: **ReLU or GELU**,
- dropout: **0.3–0.5**.

This layer reduces computational burden before temporal modeling and acts as regularization by constraining feature dimensionality.

---

## 6. BiGRU Temporal Modeling

A **BiGRU** is adopted instead of BiLSTM because it is lighter, has fewer parameters, and remains effective at capturing bidirectional temporal dependencies. This choice is better suited to small-scale video datasets such as RWF-2000.

Recommended settings:

- hidden size: **128 or 256**,
- layers: **1 or 2**,
- dropout: **0.3–0.5**,
- bidirectional: **True**.

Given the compressed frame sequence, BiGRU models temporal evolution patterns that differentiate violent from non-violent dynamics.

---

## 7. Temporal Attention Pooling

Not all sampled frames are equally informative. A temporal attention pooling layer learns frame importance weights and aggregates BiGRU outputs into a compact clip-level representation. Frames containing discriminative aggressive interactions receive higher attention, while neutral frames are down-weighted.

---

## 8. Deep Classifier Output

A lightweight binary classifier transforms the attention-pooled temporal representation into:

- \(p_{violent}\): probability of violent activity.

In addition, three auxiliary decision variables are computed:

1. **uncertainty**: entropy of class probability distribution.
2. **motion_intensity**: temporal difference magnitude between consecutive frame-level/hidden representations.
3. **temporal_consistency**: stability of frame-wise violence predictions over time.

The fuzzy system receives four inputs:

1. \(p_{violent}\)
2. uncertainty
3. motion_intensity
4. temporal_consistency

---

## 9. Interval Type-2 Fuzzy Inference System (IT2FIS)

### 9.1 Why IT2FIS (and not ANFIS)

An **Interval Type-2 FIS** is used because it explicitly handles uncertainty in ambiguous surveillance conditions, provides better interpretability than purely neural decisions, and is lighter/more controllable than combining ANFIS and Type-2 systems.

### 9.2 Fuzzy Variables

**Inputs** (all with linguistic terms Low/Medium/High):

- Input 1: \(p_{violent}\)
- Input 2: uncertainty
- Input 3: motion_intensity
- Input 4: temporal_consistency

**Output**:

- criminal_activity_risk with linguistic terms: Very Low, Low, Medium, High, Very High.

The defuzzified risk score is normalized to \([0,1]\).

### 9.3 Compact Rule Base

To prevent overfitting, the rule base is kept small (**6–12 rules**). Example rules:

1. IF \(p_{violent}\) is High AND uncertainty is Low AND temporal_consistency is High THEN risk is Very High.
2. IF \(p_{violent}\) is High AND uncertainty is High AND motion_intensity is High THEN risk is High.
3. IF \(p_{violent}\) is Medium AND motion_intensity is High AND temporal_consistency is Medium THEN risk is High.
4. IF \(p_{violent}\) is Medium AND uncertainty is High THEN risk is Medium.
5. IF \(p_{violent}\) is Low AND uncertainty is Low THEN risk is Very Low.
6. IF \(p_{violent}\) is Low AND motion_intensity is High AND uncertainty is High THEN risk is Medium.

These rules yield transparent reasoning and improve uncertain-case handling.

---

## 10. Fuzzy Q-Learning Optimizer

### 10.1 Rationale

**Fuzzy Q-Learning** is preferred over PPO because PPO is comparatively heavy for this binary decision setting and requires a more complex policy environment. Fuzzy Q-Learning is directly compatible with fuzzy decision states and can efficiently optimize thresholds/rule weights.

### 10.2 RL Formulation

**State**:
\[
s_t = [p_{violent},\ fuzzy\_risk\_score,\ uncertainty,\ motion\_intensity,\ temporal\_consistency,\ previous\_decision]
\]

**Actions**:

- \(A_0\): classify as non-violent,
- \(A_1\): classify as violent,
- \(A_2\): decrease fuzzy threshold,
- \(A_3\): increase fuzzy threshold,
- \(A_4\): increase weight of high-risk rules,
- \(A_5\): decrease weight of uncertain rules.

**Reward**:

- +2: true violent detection,
- +1: true non-violent detection,
- -1: false alarm,
- -3: missed violence,
- -0.1: unstable decision switching,
- -0.1: unnecessary uncertainty-driven delay.

Missed violence is penalized more strongly because environmental security requires prioritizing dangerous-event detection.

Fuzzy Q-Learning does **not** replace the neural classifier; it optimizes the fuzzy decision policy.

---

## 11. Final Decision Layer

Let \(\hat{r}\) be the optimized fuzzy risk score and \(\tau\) the learned threshold.

\[
\text{Decision}=
\begin{cases}
\text{Violent}, & \hat{r} \ge \tau \\
\text{Non-violent}, & \hat{r} < \tau
\end{cases}
\]

For interpretability, the system may report:

- violence probability,
- uncertainty level,
- motion intensity level,
- temporal consistency level,
- fuzzy risk score,
- activated rules.

---

## 12. Overfitting Prevention Strategy

The anti-overfitting design includes:

- ConvNeXt-Tiny (not large backbones),
- pretrained initialization,
- partial freezing of backbone,
- fine-tuning only final block + attention,
- BiGRU over BiLSTM,
- compact fuzzy rule base,
- dropout (0.3–0.5),
- AdamW with weight decay,
- label smoothing,
- early stopping,
- moderate augmentation,
- validation-based threshold tuning,
- avoiding massive end-to-end training from scratch.

---

## 13. Training Strategy

A staged strategy is adopted:

### Stage 1: Deep Spatio-Temporal Training
Train ConvNeXt-Tiny + Dual Attention + BiGRU + classifier.

- Loss: Cross-Entropy with Label Smoothing.

### Stage 2: Fuzzy Layer Training
Freeze most deep layers and train IT2FIS parameters/membership settings and initial rule weights.

### Stage 3: Fuzzy Q-Learning Optimization
Optimize fuzzy thresholds and rule-weight policies through environment interaction and reward maximization.

### Stage 4: Final Evaluation
Evaluate the integrated model on the RWF-2000 test set.

---

## 14. Suggested Hyperparameters

- Backbone: ConvNeXt-Tiny
- Frames per clip: 16 or 32
- Input size: 224×224
- Optimizer: AdamW
- LR (ConvNeXt fine-tune): \(1\times10^{-5}\)
- LR (BiGRU/attention/head): \(1\times10^{-4}\)
- Weight decay: 0.01–0.05
- Batch size: 4 or 8
- Epochs: 30–50
- Dropout: 0.3–0.5
- BiGRU hidden size: 128 or 256
- Fuzzy rules: 6–12
- RL algorithm: Fuzzy Q-Learning

---

## 15. Evaluation Metrics

The following metrics are reported:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

Because this is a security task, recall for the violent class is critical. Accuracy alone is insufficient, since missed violence carries higher operational risk than false alarms.

---

## 16. Ablation Study Design

To validate component-wise contribution:

- **Experiment A**: ConvNeXt-Tiny only
- **Experiment B**: ConvNeXt-Tiny + Dual Attention
- **Experiment C**: ConvNeXt-Tiny + Dual Attention + BiGRU
- **Experiment D**: ConvNeXt-Tiny + Dual Attention + BiGRU + IT2FIS
- **Experiment E**: ConvNeXt-Tiny + Dual Attention + BiGRU + IT2FIS + Fuzzy Q-Learning

Expected interpretation:

- Dual Attention improves spatial discriminability,
- BiGRU improves temporal modeling,
- IT2FIS improves uncertainty handling and interpretability,
- Fuzzy Q-Learning improves threshold adaptation, rule weighting, and decision stability.

---

## 17. Novelty Claim

The novelty is **not** in introducing ConvNeXt, BiGRU, fuzzy logic, or reinforcement learning independently. The contribution lies in their specific integration for lightweight, explainable violence detection on RWF-2000:

1. A lightweight fuzzy logic-based CAD framework with deep spatio-temporal features.
2. ConvNeXt-Tiny + Dual Attention integration for surveillance-oriented spatial modeling.
3. BiGRU-based temporal modeling to control parameter growth and overfitting.
4. IT2FIS for uncertainty-aware criminal activity risk inference.
5. Fuzzy Q-Learning for adaptive threshold/rule-weight optimization.
6. Joint improvement of explainability and predictive performance.

---

## 18. Final Academic Description

The proposed **RL-IT2FIS-CAD** framework is a hybrid deep neuro-fuzzy reinforcement learning architecture for criminal activity detection in surveillance videos. It first extracts discriminative spatial features from sampled video frames using a partially frozen ConvNeXt-Tiny backbone enhanced with dual attention. A BiGRU module then models temporal dependencies across frames. The resulting spatio-temporal representation is transformed into interpretable decision variables, including violence probability, uncertainty, motion intensity, and temporal consistency. These variables are passed to an Interval Type-2 Fuzzy Inference System to estimate the criminal activity risk under uncertainty. Finally, a Fuzzy Q-Learning optimizer adjusts fuzzy rule weights and decision thresholds to reduce false alarms and missed detections. The final decision is generated as violent or non-violent activity. The architecture is designed to be lightweight, explainable, and robust against overfitting on the RWF-2000 dataset.
