sequenceDiagram
    participant D as Dataset
    participant P as Processor
    participant T as Tabular Models
    participant L as LLM Models
    participant E as Evaluator
    D->>P: Load brand\_information.csv
    P->>P: Clean \& preprocess
    P->>T: Training data
    par Train in Parallel
        T->>T: Train CTGAN (300 epochs)
        T->>T: Train TVAE (300 epochs)
        T->>T: Fit Gaussian Copula
    end
    T->>T: Generate synthetic tabular
    T->>L: Tabular features
    par Generate Names
        L->>L: GPT-2 generation
        L->>L: Flan-T5 generation
    end
    L->>E: Complete synthetic data
    E->>E: Statistical tests
    E->>E: Visualization