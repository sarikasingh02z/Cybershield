**CyberShield NIDS — Research Context and Design Decisions**

**What This Project Is**
CyberShield is an experimental NIDS built to answer one specific research question:

Does confidence threshold tuning meaningfully improve detection reliability in a dual-model intrusion detection system trained on imbalanced network traffic?

The answer, validated through 5-fold cross-validation on UNSW-NB15: yes, and the improvement is reproducible and not inflated by test-set leakage.
This is not a production security system. It is a demonstration of research-grade thinking applied to a real ML problem in cybersecurity.

**Why This Problem**
Machine learning models for intrusion detection are almost universally reported with accuracy at the default threshold (θ = 0.50). This is convenient but wrong for two reasons:

-Default thresholds assume balanced classes. UNSW-NB15 training data is 55% attacks. The decision boundary at 0.50 is not optimal.
-In security contexts, the cost of a false negative (missed attack) and false positive (false alarm) are asymmetric and context-dependent. The optimal threshold depends on which cost you are minimising.

Most published work on NIDS either ignores threshold tuning entirely or mentions it as an afterthought. This project makes it the central research contribution.

**Architecture Decisions and Why**
Dual-Guard Not Competing Models
Early versions of this project ran Random Forest and Isolation Forest as parallel classifiers and compared their confusion matrices. This was the wrong framing.
RF is supervised — it was trained on labeled attacks. IF is unsupervised — it was trained on normal traffic only and has never seen an attack label. Comparing their F1 scores as if they are doing the same job is architecturally incoherent.
The correct framing is division of labour:

-Guard 1 (RF): classifies known patterns
-Guard 2 (IF): flags deviation from normal

This reframing also resolves the zero-day detection angle. When Guard 1 says NORMAL (does not recognise the pattern) but Guard 2 says the traffic is anomalous (deviates from normal), that combination is a genuine zero-day candidate signal — something neither guard alone can produce.
**Why Not SMOTE**
Class imbalance in UNSW-NB15 training data is approximately 55/45. This is moderate. SMOTE is appropriate for severe imbalance (95/5 or worse). At 55/45 it introduces synthetic samples by interpolating between real minority-class points. On network traffic data with mixed feature types (flags, byte counts, protocol identifiers), interpolated samples are not realistic traffic. They are mathematical artefacts that can distort the decision boundary.
class_weight='balanced' on Random Forest achieves the same correction mathematically without generating fake data. No SMOTE used.
**Why CV for Threshold Selection**
Sweeping thresholds on the test set and reporting the best threshold's metrics is data leakage. The threshold becomes optimised to that specific test split and the improvement claim is inflated.
The correct approach: sweep thresholds on validation folds during cross-validation, select the threshold with highest mean CV F1, then apply it to the held-out test set once. The improvement reported in this project is from that honest evaluation.
Why IF Uses contamination='auto' and decision_function()
IsolationForest's predict() method uses a threshold derived from the contamination parameter. Setting contamination to the attack ratio (0.55) causes sklearn to flag nearly half of all traffic as anomalous, producing a 47% false positive rate.
The correct approach for this architecture: use decision_function() which returns the raw anomaly score, and apply a percentile-based cutoff calibrated on the RF-normal pool. This makes IF's aggressiveness a tunable, interpretable parameter instead of a contamination hyperparameter.

**What Was Tried and Did Not Work**
IF as Binary Second-Stage Filter
Three attempts were made to use IF as a hard second-stage filter catching attacks RF missed:

-Binary predict() with contamination = attack_ratio (0.55) → 47% FP rate. Too aggressive.
-Binary predict() with contamination = RF OOB FN rate (~0.057) → Recovery ratio 0.18. For every attack recovered, 5.6 false alarms introduced.
-Score-based cutoff at p10 of RF-normal pool → Recovery ratio still below 1.0 (more false alarms than recovered attacks).

The root cause is feature space overlap. UNSW-NB15 attack traffic and normal traffic are not geometrically separated in the 15 selected features. IF's isolation principle requires anomalous points to be sparse and isolated. They are not here.
This is retained in the project as an honest documented finding, not hidden. The dual-guard architecture is kept with IF producing a score signal rather than a binary decision.
Cascade Architecture Precedes Score Architecture
The cascade architecture (IF runs only on RF-normal predictions) was tried before the dual-guard score approach. The cascade produced worse results than dual-guard scoring because it restricted IF's input to a subset, reducing the calibration surface for the percentile cutoff.

**What the Project Does Not Do**
These are intentional scope constraints:

-No live packet capture — requires root/admin access, raw socket programming, and platform-specific libraries. Outside scope of a research demo.
-No production deployment — the Gradio app is a research interface, not a deployable security tool.
-No multiclass attack classification from RF — the model is binary. Attack category labels are inferred from rule-based heuristics on packet features. A multiclass model would require a different training setup.
-No feature engineering for IF — the 15 features were selected for RF performance. A separate feature analysis optimised for IF's isolation principle (looking for features with clear distributional separation between normal and attack) could improve IF score quality. This is noted as future work.
-No real-time threshold adaptation — threshold is fixed at inference. Adaptive threshold methods exist but add complexity beyond this project's scope.


**Honest Assessment**
What works well:

-The threshold tuning finding is real and reproducible
-The dual-guard architecture is conceptually sound
-Cross-validation prevents inflated improvement claims
-The Gradio app correctly reflects the model's actual behaviour including limitations

**What is weaker than it looks:*8

-IF anomaly scores have significant overlap between normal and attack distributions on this dataset. The suspicion labels (LOW/MEDIUM/HIGH) are indicative, not reliable.
-The improvement from threshold tuning, while real and CV-validated, is modest. The larger gain would come from better feature engineering or a more sophisticated model.
-Attack category labels in the real-time detection tab are rule-based, not model-derived.

**What a reviewer should know:**
This project demonstrates understanding of ML pipeline design, evaluation methodology, and honest reporting more than it demonstrates state-of-the-art NIDS performance. The research question is answered correctly. The methodology is sound. The limitations are explicit.
