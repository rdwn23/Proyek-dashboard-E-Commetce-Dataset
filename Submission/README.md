## Setup Environment - Anaconda

```
conda create --name main-ds python=3.9
conda activate main-ds
pip install -r requirements.txt
```

## Setup Environment - Shell/Terminal

```
cd submission
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

## Run Streamlit App

```
cd dashboard
streamlit run dashboard.py
```
