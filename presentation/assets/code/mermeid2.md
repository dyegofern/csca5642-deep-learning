flowchart LR
    subgraph ENSEMBLE["Ensemble Weighting"]
        direction TB
        W1["CTGAN: 40\%"] --> MIX((Weighted<br/>Average))
        W2["TVAE: 35\%"] --> MIX
        W3["Copula: 25\%"] --> MIX
        MIX --> OUT[Synthetic Data]
    end
    style ENSEMBLE fill:\#f5f5f5
    style MIX fill:\#4caf50,color:\#fff