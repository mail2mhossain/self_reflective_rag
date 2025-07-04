# self_reflective_rag
Retrieval-Augmented Generation (RAG) pipelines are becoming essential in building intelligent applications that can retrieve and reason over large volumes of unstructured data. To make such pipelines efficient and scalable, it is important to design them with clarity, modularity, and long-term maintainability in mind.


## Setup and Running the Source Code:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mail2mhossain/self_reflective_rag.git
   cd self_reflective_rag
   ```

2. **Create a Conda environment (Assuming Anaconda is installed)**:
   ```bash
   conda create -n self_reflective_rag_env python=3.11
   ```

3. **Activate the environment**:
   ```bash
   conda activate self_reflective_rag_env
   ```

4. **Install the required packages**:
   ```bash
   pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install git+https://github.com/huggingface/transformers
   pip install git+https://github.com/huggingface/accelerate
   
   pip install -r requirements.txt
   ```

5. **Run the data ingestion pipeline**:
   ```bash
   python -m rag.data_ingest_pipeline.pipeline [from root directory]
   ```

6. **Run the retrieval pipeline**:
   ```bash
   python -m rag.data_retrieval.pipeline [from root directory]
   ```

7. **Run the generation pipeline**:
   ```bash
   python -m rag.self_rag.main [from root directory]
   ```

*To remove the environment after use:*
```bash
conda remove --name self_reflective_rag_env --all
```
