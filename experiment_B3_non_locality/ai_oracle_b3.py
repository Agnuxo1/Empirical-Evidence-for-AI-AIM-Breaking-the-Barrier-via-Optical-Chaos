import numpy as np
import os
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

def run_ai_oracle():
    print("=== Experiment B3: AI Oracle (The Non-Local Link) ===")
    print("Objective: Predict Outcome B given (Axis A, Axis B, Outcome A).")
    print("Constraint: Local Realism implies Accuracy <= ~75% (Bell Limit).")
    print("Hypothesis: If AI Accuracy > 75%, it has learned Non-Locality.")
    
    # 1. Load Data
    data_path = "entanglement_data.npy"
    if not os.path.exists(data_path):
        print("Error: Data file not found.")
        return
        
    data = np.load(data_path)
    # Data: [axis_A, axis_B, outcome_A, outcome_B]
    print(f"Loaded Data Shape: {data.shape}")
    
    # 2. Prepare Inputs (X) and Targets (y)
    # Input: Axis A, Axis B, Outcome A
    # Target: Outcome B
    X = data[:, 0:3]
    y = data[:, 3]
    
    # Split Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Train AI (Neural Network)
    # We use a small MLP. It's a universal approximator.
    # If there is a function f(A, B, OutA) -> OutB, it should find it.
    # In QM, the function is probabilistic: P(OutB | ...)
    # But for specific angles (0, pi), it is deterministic.
    # The AI should learn to be confident when angles are aligned/anti-aligned, and guess when orthogonal.
    
    print("Training AI Oracle (MLPClassifier)...")
    clf = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    
    # 4. Evaluate
    accuracy = clf.score(X_test, y_test)
    print(f"AI Prediction Accuracy: {accuracy*100:.2f}%")
    
    # 5. Analysis
    # Bell's Inequality (CHSH) roughly translates to a max correlation of 2.
    # In terms of prediction accuracy for optimal angles, Classical Max is ~75%.
    # Quantum Max is ~85% (cos^2(pi/8) approx).
    # Let's check accuracy on "easy" subset (aligned axes) vs "hard" subset.
    
    print("\nDetailed Analysis:")
    
    # Check aligned axes (diff < 0.5 rad)
    diffs = np.abs(X_test[:, 0] - X_test[:, 1])
    diffs = np.minimum(diffs, 2*np.pi - diffs)
    mask_aligned = diffs < 0.5
    
    if np.any(mask_aligned):
        acc_aligned = clf.score(X_test[mask_aligned], y_test[mask_aligned])
        print(f"Accuracy on Aligned Axes (<0.5 rad): {acc_aligned*100:.2f}%")
    
    # Check orthogonal axes (diff ~ pi/2)
    mask_ortho = np.abs(diffs - np.pi/2) < 0.5
    if np.any(mask_ortho):
        acc_ortho = clf.score(X_test[mask_ortho], y_test[mask_ortho])
        print(f"Accuracy on Orthogonal Axes (~pi/2): {acc_ortho*100:.2f}% (Should be ~50%)")
        
    if accuracy > 0.75:
        print("\nConclusion:")
        print("The AI Oracle achieved > 75% accuracy.")
        print("Cage Status: BROKEN (Information Non-Locality).")
        print("The AI effectively learned the Quantum Correlation function.")
    else:
        print("The AI failed to beat the Classical Limit.")

if __name__ == "__main__":
    run_ai_oracle()
