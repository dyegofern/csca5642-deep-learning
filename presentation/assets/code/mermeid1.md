flowchart TB
    subgraph INPUT["Input Data"]
        A[(Brand Dataset<br/>CSV)] --> B[Data Processor]
    end
    subgraph PREPROCESS["Preprocessing"]
        B --> C[Clean \& Validate]
        C --> D[Feature Engineering]
        D --> E[Train/Test Split]
    end
    subgraph TABULAR["Tabular Data Generation"]
        E --> F1[CTGAN<br/>Conditional GAN]
        E --> F2[TVAE<br/>Variational Autoencoder]
        E --> F3[Gaussian Copula<br/>Statistical Model]
        F1 --> G[Ensemble<br/>Weighted Averaging]
        F2 --> G
        F3 --> G
    end
    subgraph TEXT["Text Generation"]
        G --> H1[GPT-2 Medium<br/>Fine-tuned LLM]
        G --> H2[Flan-T5 Small<br/>Instruction-tuned]
        H1 --> I[Text Ensemble<br/>Best Selection]
        H2 --> I
    end
    subgraph OUTPUT["Output"]
        I --> J[Synthetic Brands<br/>with Names]
        J --> K[Quality Evaluation]
    end
    subgraph EVAL["Evaluation Metrics"]
        K --> L1[KS Test]
        K --> L2[Correlation]
        K --> L3[PCA/t-SNE]
        K --> L4[Clustering]
    end
    style INPUT fill:\#e1f5fe
    style PREPROCESS fill:\#fff3e0
    style TABULAR fill:\#f3e5f5
    style TEXT fill:\#e8f5e9
    style OUTPUT fill:\#fce4ec
    style EVAL fill:\#fff8e1