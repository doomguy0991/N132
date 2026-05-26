import subprocess
import os
import json

# 1. Define Mermaid Diagram Codes
diagrams = {
    "lecture3_diagram_1.mmd": """graph TD
    classDef default fill:#1e1e2e,stroke:#cba6f7,stroke-width:2px,color:#cdd6f4;
    classDef loss fill:#f38ba8,stroke:#f38ba8,stroke-width:2px,color:#11111b;
    classDef reg fill:#89b4fa,stroke:#89b4fa,stroke-width:2px,color:#11111b;
    classDef hyper fill:#fab387,stroke:#fab387,stroke-width:2px,color:#11111b;
    classDef model fill:#a6e3a1,stroke:#a6e3a1,stroke-width:2px,color:#11111b;

    A[Input Data & Labels] --> B(Data Loss L_data):::loss
    C[Model Weights W] --> D(Regularization Loss R W):::reg
    
    B -->|Tug 1: Fit training data closely| E(Joint Objective L_total):::model
    D -->|Tug 2: Penalize model complexity| E
    
    F(Hyperparameter lambda):::hyper -->|Controls regularization strength| E
    
    E -->|Output| G[Simpler & More Robust Weights]
""",
    "lecture3_diagram_2.mmd": """graph TD
    classDef default fill:#1e1e2e,stroke:#cba6f7,stroke-width:2px,color:#cdd6f4;
    classDef num fill:#fab387,stroke:#fab387,stroke-width:2px,color:#11111b;
    classDef ana fill:#a6e3a1,stroke:#a6e3a1,stroke-width:2px,color:#11111b;
    classDef check fill:#89b4fa,stroke:#89b4fa,stroke-width:2px,color:#11111b;

    W[Current Weights W] --> Choice{Choose Gradient Method}
    
    Choice -->|Finite Differences W+h| Num[Numerical Gradient]:::num
    Choice -->|Calculus & Chain Rule| Ana[Analytic Gradient]:::ana
    
    Num -->|Approximate & Slow| Check[Gradient Check]:::check
    Ana -->|Exact & Fast| Check
    
    Check -->|Compare Relative Error| Verify{Is Error < 1e-7?}
    Verify -->|Yes| Good[Safe to Train Model]
    Verify -->|No| Bad[Debug Analytic Derivation]
""",
    "lecture3_diagram_3.mmd": """graph TD
    classDef default fill:#1e1e2e,stroke:#cba6f7,stroke-width:2px,color:#cdd6f4;
    classDef base fill:#f38ba8,stroke:#f38ba8,stroke-width:2px,color:#11111b;
    classDef mom fill:#fab387,stroke:#fab387,stroke-width:2px,color:#11111b;
    classDef adapt fill:#89b4fa,stroke:#89b4fa,stroke-width:2px,color:#11111b;
    classDef final fill:#a6e3a1,stroke:#a6e3a1,stroke-width:2px,color:#11111b;

    SGD[Vanilla SGD]:::base -->|Accumulate past steps with momentum| SGD_M[SGD with Momentum]:::mom
    SGD -->|Adapt step sizes by squared gradients| RMS[RMSProp]:::adapt
    
    SGD_M -->|Combine direction + adaptive scaling| Adam[Adam Optimizer]:::final
    RMS -->|Combine direction + adaptive scaling| Adam
    
    Adam -->|Decouple L2 weight decay from moments| AdamW[AdamW Optimizer]:::final
"""
}

# Ensure assets folder exists
os.makedirs("assets", exist_ok=True)

# 2. Write MMD files and compile to PNG using global mmdc
for filename, code in diagrams.items():
    png_filename = filename.replace(".mmd", ".png")
    png_path = os.path.join("assets", png_filename)
    
    # Write code to temp mmd file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(code)
    
    print(f"Compiling {filename} to {png_path}...")
    # Execute mmdc command
    try:
        subprocess.run(["mmdc", "-i", filename, "-o", png_path, "-b", "transparent"], check=True, shell=True)
        print(f"Successfully compiled {png_path}!")
    except Exception as e:
        print(f"Failed to compile {filename}: {e}")
        
    # Clean up temp file
    if os.path.exists(filename):
        os.remove(filename)

# 3. Load CS231n_Lecture3_Study_Notes.ipynb and inject the diagrams
with open("CS231n_Lecture3_Study_Notes.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

# We will locate the target markdown cells and append the markdown images to their sources.
for cell in notebook["cells"]:
    if cell["cell_type"] == "markdown":
        source_str = "".join(cell["source"])
        
        # Injection 1: Section 1.3 Tug-of-War
        if "### The Tug-of-War Intuition" in source_str and "assets/lecture3_diagram_1.png" not in source_str:
            # We insert the diagram right after "The Tug-of-War Intuition" header or before the bullet list
            print("Injecting Diagram 1 into Section 1.3...")
            cell["source"].append("\n\n### Pipeline / Objective Balance Diagram\n\n![Diagram 1](assets/lecture3_diagram_1.png)\n")
            
        # Injection 2: Section 2.4 Numerical vs. Analytic Gradients
        if "## 2.4 Numerical vs. Analytic Gradients" in source_str and "assets/lecture3_diagram_2.png" not in source_str:
            print("Injecting Diagram 2 into Section 2.4...")
            cell["source"].append("\n\n### Gradient Computation & Checking Flow\n\n![Diagram 2](assets/lecture3_diagram_2.png)\n")
            
        # Injection 3: Section 3.7 Decoupled Weight Decay (Adam vs AdamW) OR Section 3.6 Adam
        if "## 3.7 Decoupled Weight Decay: Adam vs. AdamW" in source_str and "assets/lecture3_diagram_3.png" not in source_str:
            print("Injecting Diagram 3 into Section 3.7...")
            cell["source"].append("\n\n### Taxonomy & Evolution of Optimizers\n\n![Diagram 3](assets/lecture3_diagram_3.png)\n")

# Save the updated notebook
with open("CS231n_Lecture3_Study_Notes.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=4)

print("Notebook updated successfully with visual diagrams!")
