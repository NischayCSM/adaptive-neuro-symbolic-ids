# Continual Learning Experience Split for UNSW-NB15

To properly evaluate our Adaptive Neuro-Symbolic IDS in a continual learning context, we need to simulate a stream of data where new attack types emerge over time.

## Objective
Split the UNSW-NB15 dataset into 4 progressive "Experiences" while maintaining a reasonable balance of training records across the splits so the model doesn't overfit on early experiences and abruptly starve on later ones.

## Chosen Configuration
Based on the standard partitioning of the UNSW-NB15 dataset (approx ~175,000 total training records), the dataset is highly imbalanced, heavily favoring Normal, Generic, and Exploits. 

We have chosen the following sequential split:

1. **Experience 1 (The Baseline):** `Normal`, `DoS`
   - *Estimated Size:* ~68,000 records
   - *Rationale:* Establishes the core baseline of distinguishing heavily represented normal traffic from a standard, well-known volumetric attack (Denial of Service).
   
2. **Experience 2 (The Noise):** `Generic`, `Reconnaissance`
   - *Estimated Size:* ~50,000 records
   - *Rationale:* Introduces high-volume generic attacks and reconnaissance sweeps, forcing the model to adapt to new structural noise without forgetting the core DoS signatures.

3. **Experience 3 (The Intrusions):** `Exploits`, `Analysis`, `Backdoors`
   - *Estimated Size:* ~37,000 records
   - *Rationale:* Groups together targeted, complex payload attacks. This is a critical point where the Neuro-Symbolic fusion should excel at locking in specific exploit thresholds.

4. **Experience 4 (The Minority Classes):** `Fuzzers`, `Shellcode`, `Worms`
   - *Estimated Size:* ~19,000 records
   - *Rationale:* Introduces the rarest, most highly imbalanced attack categories at the end of training. This will severely test the model's ability to extract and memorize rules for minority classes (like Worms, which has < 200 records) without catastrophic forgetting of the previous 150k records.

## Usage
This split is configured in `configs/continual.yaml` and is consumed dynamically by the `ExperienceBuilder` module in `src/continual/experience_builder.py`.
