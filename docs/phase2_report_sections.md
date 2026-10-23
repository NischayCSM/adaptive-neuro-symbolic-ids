*Drafting note (delete before pasting): every result number below is taken from `results/PHASE2_RESULTS_v2.md`, whose
"Number sources" table gives the results file and exact value for each. Values are mean ± sample standard
deviation over five seeds unless stated otherwise. Figure numbers F1–F6 refer to `results/figures_phase2/`.*

# Phase II Methodology

Phase II revisits the evaluation protocol of Phase I before extending the system. It makes four protocol
changes, adds two methodological components (a learned fusion layer and a revised rule-validation criterion), and
adds an external evaluation on harmonised datasets.

### 1. Final-model evaluation on the full test set
In Phase I, continual performance was summarised by the accuracy of the model trained up to experience *k*,
measured on the test split of experience *k*. This quantity is the diagonal of the accuracy matrix. It measures
plasticity rather than retention, and it cannot reveal a model that forgets earlier classes. In Phase II, the
model obtained after the final experience is evaluated once on the complete ten-class test set (175,341 flows,
equal to the union of the four experience test splits). Macro-F1 is the primary metric because the class
distribution is highly imbalanced. To diagnose bias towards the most recently learned classes, we also report the
*recency share*: the fraction of all predictions assigned to the classes of the final experience (Fuzzers,
Shellcode and Worms), which should be compared with their true share of 0.1109.

### 2. Separation of calibration and evaluation data
Fusion hyperparameters were previously chosen and reported on the same test data. In Phase II the test set is
divided once, by stratified sampling, into a calibration split (35,069 flows, 20%) and an evaluation split
(140,272 flows, 80%). Every fusion result is reported on the evaluation split. In the static setting, differences
from the detector alone carry 95% paired-bootstrap confidence intervals.

### 3. Validation memory
Calibrating a fusion layer on labelled examples of all ten classes is a post-hoc assumption: a deployed continual
system has no labelled data for classes it has not yet encountered. Phase II therefore introduces a *validation
memory*. Before each experience is learned, up to 200 training flows per class of that experience (at most half of
the class) are withheld from training and from the replay buffer. All fusion weights and stacking models are
fitted on this memory alone. Each seed's memory holds 1,811 flows, alongside the 10,000-flow replay buffer.

### 4. Leakage-free knowledge base
Because the validation memory is drawn from the training data, rules induced from that same data could have been
validated on the flows later used to fit the fusion layer. The final knowledge bases are therefore rebuilt with the
union of all seeds' validation-memory flows (7,167 flows) excluded from rule induction and validation.

### 5. Rule-feature stacking and a detector-only control
Phase I combined the detector and the rules by a fixed linear blend, $\alpha P + (1-\alpha) S$, where $S$ is the
highest stored precision among the fired rules of each class (V0). Phase II evaluates three hand-designed
refinements: confidence gating (V1); neutral absence, in which a class without a fired rule is scored by the
detector alone (V2); and their combination (V3). It also evaluates a learned alternative, V4: a class-balanced
multinomial logistic regression over the detector's log-probabilities and, for each class, the highest precision
of the fired rules, a fired indicator and the logarithm of the number of fired rules. Any gain from such a stacker
could come from recalibrating the detector rather than from the rules. We therefore fit an identical model on the
detector's log-probabilities alone (the *control*). We define the **rule contribution** as the macro-F1 of V4 minus
that of the control, with the same detector and the same calibration data.

### 6. Rule validation by normalised precision gain
The Phase I validator accepts a rule whose precision on a validation half is at least 0.85. Under class imbalance,
a rule that matches nearly every flow inherits the base rate of the dominant class as its precision and passes
this test while carrying no information. Phase II evaluates two additional criteria, each combined with a cap of
0.5 on the fraction of validation flows a rule may match. The first is lift (precision divided by the class base
rate, at least 1.5). The second is normalised precision gain,
$g = (\text{precision} - b)/(1 - b)$, where $b$ is the class base rate, with $g \ge 0.5$. Unlike lift, whose maximum
$1/b$ approaches 1 for a dominant class, gain can reach 1 for every class.

### 7. Stronger continual detectors
Two standard remedies for class-incremental bias are added to the replay detector: inverse-frequency class
weighting of the cross-entropy loss over each experience's training mixture (R1), and Weight Aligning of the output
layer after each experience (R2; Zhao et al., CVPR 2020). R3 combines both, and R0 denotes the unmodified
replay detector. The **final configuration** is R1 with replay buffer 10,000 and the validation memory, the
gain-validated leakage-free knowledge base (90 active rules), and V4 fitted on the validation memory.

### 8. External validation on harmonised NetFlow features
The Phase I transfer experiment to TON_IoT required hand-mapping six features and zero-filling the rest, which
confounds feature mismatch with genuine distribution shift. Phase II instead uses NF-UNSW-NB15-v2 and
NF-ToN-IoT-v2, which share a common NetFlow feature set (Sarhan et al., arXiv 2101.11315). Models and rules are
trained on NF-UNSW-NB15-v2 only. The fusion layer is fitted on its validation split, and evaluation is performed
both in-domain and on NF-ToN-IoT-v2 using the same 39 features. Both datasets were subsampled to at most 50,000
flows per class, so pooled rates are comparable only within this experiment.

# Results and Discussion

### Catastrophic forgetting: fine-tuning, EWC and replay
Under the final-model protocol, naive fine-tuning and Elastic Weight Consolidation (EWC; Kirkpatrick et al., 2017)
both collapse onto the last experience (Figure F1). Their final models assign almost every test flow to the
Experience 4 classes: the recency share is 0.9895 ± 0.0091 for naive fine-tuning and 1.0000 for EWC at
$\lambda \ge 10^5$, against a true share of 0.1109. Macro-F1 is 0.0248 ± 0.0053 for naive fine-tuning and at most
0.0224 ± 0.0060 for EWC across the four regularisation strengths tested. In this class-incremental setting,
parameter regularisation alone therefore did not stop the output layer from favouring the most recent classes,
which never co-occur with earlier classes in a training batch. Experience replay behaves very differently. Final-model macro-F1
increases with the buffer size, from 0.2632 ± 0.0139 (2,000 flows) to 0.3035 ± 0.0187 (5,000) and
0.3751 ± 0.0052 (10,000), and the recency share falls from 0.4525 to 0.2185. With 10,000 stored flows, replay lies
within 0.0116 macro-F1 of joint training for 20 epochs (0.3867 ± 0.0111). These results revise the central claim
of Phase I. The reported 90–94% "retention" was a diagonal average that cannot detect forgetting. On the
full-test protocol, the protection against forgetting observed here is attributable to replay. The symbolic layer
was not evaluated as a substitute for replay.

### Class weighting closes most of the gap to joint training
Class-weighted replay (R1) raises final-model macro-F1 to 0.4005 ± 0.0139 on the full test set. The class-weighted
joint-training upper bound is 0.4079 ± 0.0162, which R1 approaches but does not reach: it exceeds the joint mean in
two of five seeds. The improvement comes from the rare classes. Without class weighting, the validation-memory
replay detector (R0) never correctly identifies Analysis, Backdoor, Shellcode or Worms (recall 0.0000 for each).
R1 recovers recalls of 0.2245, 0.0779, 0.4901 and 0.5365 respectively. This comes at the cost of lower recall for
DoS (0.4948 → 0.3752), Fuzzers (0.8656 → 0.5823) and Reconnaissance (0.5985 → 0.4314) (evaluation split). Weight Aligning (R2) yields a
smaller gain (detector macro-F1 0.3800 ± 0.0038, against 0.3680 ± 0.0073 for R0).

### In the continual setting, rule-feature stacking adds a gain beyond recalibration
In the static setting, no fusion variant improves on the XGBoost detector. With the static knowledge base, the
Phase I fusion (V0, $\alpha = 0.5$) changes evaluation macro-F1 by −0.0185 [−0.0204, −0.0168], and no variant has a
confidence interval above zero. In the continual setting, by contrast, V4 has the highest mean macro-F1 in all nine
test-calibrated replay and knowledge-base combinations. The control experiment indicates that this gain is not
explained by recalibration alone: with the same detector and the same calibration data, adding the rule features
raises macro-F1 by
+0.0287 ± 0.0072 (R0), +0.0346 ± 0.0157 (R1), +0.0315 ± 0.0049 (R2) and +0.0435 ± 0.0166 (R3). The rule
contribution is positive in five of five seeds for every detector (Figure F2). It also remains positive in five of
five seeds when the feature ct_state_ttl, which the preprocessing code flags as a potential label leak, is removed
(+0.0374), and when the whole TTL feature group is removed (+0.0485). Under the legal, leakage-free protocol, the final configuration reaches a macro-F1 of
0.4303 ± 0.0049 on the evaluation split, against 0.4008 ± 0.0136 for its detector alone and 0.4081 ± 0.0162 for
class-weighted joint training. With test calibration, V4 reached 0.4456 ± 0.0022 (replay-10,000 detector, Phase
I-style knowledge base). Under the legal, leakage-free protocol it reached 0.4320 ± 0.0057. The two settings also
differ in how the detector was trained (validation memory held out) and in the knowledge base, so this
difference bounds, rather than isolates, the optimism of test calibration.

The stacker mainly redistributes errors rather than raising accuracy uniformly (Figure F5). For the final
configuration, V4 increases recall for DoS (0.3752 → 0.4387), Fuzzers (0.5823 → 0.6523), Reconnaissance
(0.4314 → 0.5676), Shellcode (0.4901 → 0.6406) and Worms (0.5365 → 0.8712). It reduces recall for Analysis
(0.2245 → 0.2056), Backdoor (0.0779 → 0.0331) and Exploits (0.4209 → 0.3887). The stacker does not make Backdoor
detectable. Across the four detector variants, V4's Backdoor recall lies between 0.0331 and 0.0601. It is lower
than with the detector alone for both class-weighted detectors (R1: 0.0779 → 0.0331; R3: 0.1384 → 0.0331), and
higher only where the detector almost never identifies Backdoor (R0: 0.0000 → 0.0503; R2: 0.0006 → 0.0601).
Accuracy changes by much less than macro-F1, and its sign depends on the detector: +0.0125 for R1, +0.0007 for R0, and −0.0093 for the
test-calibrated replay-10,000 detector. The fixed linear blend of Phase I, in contrast, can be harmful. With a
knowledge base that lacks Fuzzers rules, V0 reduces Fuzzers recall from 0.8325 to 0.0009, because high-precision
rules for other classes outvote the detector on Fuzzers traffic.

### Benign false alarms and missed attacks
The macro-averaged false-positive rate used in the Phase I scorecard averages the Normal class with nine attack
classes and therefore does not show how many benign flows raise an alarm. We therefore also report the
Normal→attack rate (the share of benign flows assigned to any attack class) and the attack→Normal rate (the share
of attacks assigned to Normal) on the evaluation split.

The two families of systems err in opposite directions:
- The static XGBoost detector flags 0.0168 of benign flows (0.0179 with V2 fusion) but misses 0.1429 of attacks
  (0.1407 with V2).
- The final continual configuration flags 0.1873 ± 0.0081 of benign flows and misses only 0.0176 ± 0.0050 of
  attacks.

The high benign false-alarm rate is already present in the continual detector before rules are added (R1 alone:
0.1827 ± 0.0067), and the stacker leaves it essentially unchanged while lowering the miss rate from
0.0286 ± 0.0076. For the same predictions, the macro-averaged false-positive rate is only 0.0328 ± 0.0004. A
scorecard based on that metric would therefore pass a configuration that raises an alarm on almost one benign
flow in five. Scored by the Normal→attack rate with a target of 0.05, the static configuration passes and the
continual configuration fails.

The high false-alarm rate is a property of the default operating point rather than of the ranking (Figure F7). As
detectors of attack versus Normal, the continual models rank flows well:
- ROC-AUC is 0.9760 ± 0.0019 for the detector alone, 0.9783 ± 0.0009 for the control and 0.9800 ± 0.0006 for V4.
- Static XGBoost reaches 0.9861.

We therefore set an alarm threshold on the score 1 − P(Normal), chosen per seed on the validation memory alone so
that at most 5% of its benign flows raise an alarm. On the evaluation split, V4 then flags 0.0347 ± 0.0059 of
benign flows, within the target in five of five seeds. The cost is a higher miss rate, 0.1518 ± 0.0138, comparable
to the 0.1429 of static XGBoost, and a lower macro-F1 (0.3807 ± 0.0062). The largest loss is on Fuzzers, whose
detection rate falls from 0.9454 to 0.2967. Every other attack class keeps a detection rate of at least 0.7442.

Because V4 is fitted on the same memory, this threshold is set on its training rows. A cross-fitted threshold
(five folds within the memory) gives 0.0278 ± 0.0073, also within the target in five of five seeds. A 1% target is
not met reliably: V4 realises 0.0151 ± 0.0043 and stays within it in no seed. With 200 benign flows per memory,
such a threshold rests on only two permitted alarms.

![Figure F7](../results/figures_phase2/F7_roc.png)

*Figure F7. ROC of attack versus Normal on the evaluation split (curves: seed 42; ROC-AUC: mean ± SD over five
seeds), with V4's validation-chosen 5% and 1% operating points.*

### Absolute rule precision is misleading under class imbalance
The precision floor of Phase I admits rules that are uninformative. In the Phase I-style knowledge base (182 active
rules), 45 rules fire on more than half of the test set, and the mean test lift is 1.90. One example is
`is_sm_ips_ports <= 4.661 → Normal`. It has a stored precision of 0.857 but fires on 0.984 of test flows, with a
test precision of 0.308, below the Normal base rate of 0.319. A lift criterion removes such rules (26 rules, 3
broad, mean lift 2.63), but it also leaves dominant classes without rules, and legal V4 falls from 0.4404 to
0.4239. The normalised-gain criterion retains rules for dominant classes while still rejecting broad ones (Figure
F3). The leakage-free gain-validated knowledge base has 90 rules, of which 3 are broad. Its mean test lift is 1.97
and its mean test gain +0.168, against 206 rules, 53 broad, 1.72 and +0.155 without the gain check. V4 reaches the
same macro-F1 with either base (0.4320 ± 0.0057 against 0.4315 ± 0.0057), so the gain criterion reaches the same
V4 macro-F1 with fewer than half of the rules. Rule quality should therefore be judged relative to the class base
rate, not by absolute precision.

### Label-free drift detection can miss new attack classes
Rule induction is triggered when the mean two-sample Kolmogorov–Smirnov statistic between a new experience and the
reference data exceeds a threshold. Experiences 2 and 3 produce clear scores (0.5427 ± 0.0050 and
0.5397 ± 0.0061). Experience 4, which introduces Fuzzers, Shellcode and Worms, scores only 0.1845 ± 0.0029. At the
default threshold of 0.15 this experience is flagged in all five seeds; at 0.20 or 0.25 it is flagged in none
(Figure F4). The knowledge base then contains no Fuzzers rules (16 → 0), V4 macro-F1 falls from 0.4303 to 0.4174,
and the rule contribution falls from +0.0346 to +0.0216. A drift score that averages per-feature marginal shifts,
and that uses no labels, can therefore under-react to new attack classes whose per-feature marginal distributions
differ little from the reference data. In this experiment, the experience that introduced three new classes, and
therefore most needed new rules, produced the weakest drift signal.

### External validation on NetFlow-v2
In-domain on NF-UNSW-NB15-v2, XGBoost already separates attacks from benign traffic almost perfectly (detection
rate 0.9996 ± 0.0001 at a Benign false-positive rate of 0.0050 ± 0.0001), and the rules add no measurable
multi-class performance. Macro-F1 is 0.6471 ± 0.0008 for XGBoost, 0.6644 ± 0.0022 for the control and 0.6637 ± 0.0025 for V4,
a rule contribution of −0.0008 ± 0.0013. The validated NetFlow knowledge base contains 16 rules and none for
Analysis, Backdoor, DoS, Shellcode or Worms. These are the classes on which XGBoost is weakest (recall 0.1449,
0.0767, 0.3390, 0.7336 and 0.7083), so the knowledge base offers no rules for the classes where the detector is
weakest. Across datasets, neither component transfers (Figure F6). On NF-ToN-IoT-v2, XGBoost flags
0.6425 ± 0.0296 of attacks but also 0.8725 ± 0.0098 of benign flows. V4 raises detection to 0.7812 ± 0.0055, but
it also flags 0.9512 ± 0.0067 of benign flows. On the four classes with a counterpart in the source data, XGBoost reaches a macro-F1 of only
0.0424 ± 0.0063. Support-weighted rule precision falls from 0.991 on source validation data to 0.182 on mapped
target flows. Because both datasets share the same feature definitions, the failure is not explained by a
mismatch of feature schemas. It is consistent with concept shift between the two networks and with the limited
generalisability of ML-based NIDS reported by Layeghy et al. (arXiv 2205.04112).

# Limitations and Future Work

The present study has several limitations that motivate future work.

1. **Labelled calibration memory.** The stacker requires a labelled validation memory for every class encountered
   so far. For the rarest class (Worms) this memory holds only 22 flows per seed. Few-shot adaptation methods that
   update the fusion layer, or the detector, from a handful of labelled target flows would reduce this
   requirement. They are also a natural route to cross-dataset deployment: in the source–target pair studied in
   Phase II, neither the detector nor the rules transferred without target data.
2. **Stacker capacity and interpretability.** V4 is a linear model over rule features. Stronger stackers (for
   example, gradient-boosted or attention-based combiners over rule activations) may capture interactions between
   rules. However, they could further weaken the link between a prediction and an individual rule: at present,
   0.8218 of V4's predictions are supported by a fired rule for the predicted class.
3. **Drift detection.** The mean-KS detector is label-free and marginal, and it under-reacted to the experience
   that introduced three new attack classes. Class-aware novelty detection, for example density- or
   prototype-based scores in the detector's representation space, could target the event that actually requires
   new rules: the appearance of an unseen class.
4. **Rules for rare classes.** No continual knowledge base contains rules for Analysis or Backdoor, and their
   recall remains below 0.21 for every detector variant. Rule induction targeted at rare classes, for example
   class-balanced candidate generation or per-class support thresholds, is likely to be needed before rules can
   help where the detector is weakest.
5. **False alarms in the continual system.** The final continual configuration labels 0.1873 ± 0.0081 of Normal
   flows as attacks, against 0.0179 for the static detector, and fails a 0.05 false-alarm target. The rate
   originates in the continual detector (0.1827 ± 0.0067 without rules) and is not reduced by the stacker. The
   macro-averaged false-positive rate (0.0328 ± 0.0004 for the same predictions) hides it, so operational
   evaluation should report the benign false-alarm rate directly. A threshold on 1 − P(Normal) chosen on the
   validation memory brings it to 0.0347 ± 0.0059, but it raises the miss rate from 0.0176 ± 0.0050 to
   0.1518 ± 0.0138. Lower targets (1%) cannot be set reliably from 200 benign flows per memory.
6. **Experimental scope.** The knowledge base was induced with a single pipeline seed. External validation uses one
   source and one target dataset with rebalanced sampling. The UNSW-NB15 partition used throughout is the reverse
   of the official one (see Errata).

# Errata for the Phase I Report

1. **Eq. 3.1** omitted the precision term. The rule score is
   $\text{Score} = 0.4\,P + 0.3\,\min(C, 0.5) + 0.2\,T - 0.2\,K$, where $P$ is precision, $C$ coverage, $T$
   stability and $K$ the conflict indicator.
2. **Stability $T$** is the precision of the rule on the second validation half (correct matches divided by
   matched flows; 0 if the rule matches no flow there). It is not a cross-validation statistic.
3. **The rule status field** takes six values, not five: CANDIDATE, VALIDATED, ACTIVE, DEGRADED, UPDATED and
   RETIRED.
4. **TON_IoT sample size:** 50,000 flows were sampled from the TON_IoT network file (random state 42), and the
   first 25,000 of these were evaluated. The two figures refer to these two steps.
5. **Section 3.4 baselines:** RandomForest, XGBoost and the MLP baselines were trained on the binary
   normal-versus-attack task. A separate 10-class XGBoost model (`xgboost_multi`) was used for fusion. NeuralIDS was
   trained for 10 epochs with batch size 1,024 in the static setting, and for 5 epochs per experience with batch
   size 64 in the continual, replay and EWC settings. Joint training used batch size 64 with 5 or 20 epochs.
6. **Section 3.6:** the knowledge base is an SQLite database in which each rule's conditions are stored as JSON
   text, not a "JSON-based database".
7. **Section 3.3:** rule thresholds on numeric features are in standardised units (z-scores from the
   training-set StandardScaler). Thresholds on proto, state and service are ordinal category codes. Neither is in
   raw units.
8. **Test 5 "PASS 0.9449"** was the mean of the continual neuro-symbolic accuracies on Experiences 1–3, each
   measured when that experience was learned, (0.9563 + 0.9793 + 0.8991) / 3 (Section 4, scorecard discussion).
   This is an average of the diagonal of the accuracy matrix, not retention. Final-model retention of the Phase I replay model (buffer 2,000, seed 42, mean
   accuracy on Experiences 1–3 after Experience 4) is 0.5514, which fails the 0.80 target; the corrected Phase I
   scorecard is 5/8.
9. **The drift flag never gated rule induction.** `DriftDetector.detect` returns a dictionary, and the pipeline
   tested that dictionary's truthiness, which is always true. Rule induction therefore ran at every experience
   regardless of the drift score. In Phase I all logged scores (0.5214, 0.4845, 0.1592) exceeded the 0.15
   threshold, so the reported knowledge bases are unaffected.
10. **Dataset partition:** in our copy of UNSW-NB15, the file named `UNSW_NB15_training-set.csv` contains 82,332
    flows and `UNSW_NB15_testing-set.csv` contains 175,341, the reverse of the official partition. All experiments
    train on the 82,332-flow file and test on the 175,341-flow file. The results are internally valid but are not
    directly comparable with studies that use the official partition.
