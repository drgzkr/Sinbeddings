# Sinbeddings

See the sin trajectory of any sequence of events.
Given a text with multiple sentences, an LLM codes each sentence based on how much of which 7 sin is present. 
The dashboard then visualizes the sin journey of the text with an animated radar plot and state space trajectory made up of 2 principal components of the whole sinbedding timeline.
The dimensions are also interpreted based on which sin is loaded in them to characterize a unique sin-space for the text.

Play with it by giving it biographies, or known literary texts, or your own stories.

![[/readmeimages/readme1.png|500]]

![[/readmeimages/readme2.png|500]]

---

## Setup

- Step 1: Create and go to the project directory
```bash
# Create folder for the project
mkdir Sinbeddings
# Move to the directory
cd Sinbeddings
```
- Step 2: Create virtual environment
```bash
python3 -m venv tts_env
source sinbedding_env/bin/activate
```
- Step 3: Install dependencies
```bash
pip install streamlit
pip install openai
```


---
## Usage
- Step 1: Open the terminal app
- Step 2: Move to the project path and activate the virtual environment
``` bash
# Move to the directory
cd Sinbeddings
# activate environment
sinbedding_env/Scripts/Activate.ps1     
```
- Step 3: Make sure the streamlit python script is at this directory
- Step 4: Figure out where to put your openai API key for the LLM sin-coding to work.
- Step 5: Run streamlit script
```bash
# run python script
streamlit run sinbedding_dashboard_v4.py   
```

