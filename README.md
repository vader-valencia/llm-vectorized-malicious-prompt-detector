# llm-vectorized-malicious-prompt-detector
This project is a proof-of-concept, black box detector for few-or-many-shot jailbreaking in LLMs using embeddings of the prompt. It compares incoming prompts against a database of known vulnerabilities. If similarity exceeds a set threshold, the prompt is blocked, enhancing LLM security.

# Running the application (Tuned for Windows for the Proof-of-Concept)
1. Install the dependencies: 
```
pip install openai --upgrade
pip install flask requests python-dotenv
```

2. 

2. Start the app
```
python -m flask --app run.py run
```

# Acknowledgements
The base of this repo was developed from [abhinav-upadhyay's](https://github.com/abhinav-upadhyay) [chatgpt_plugins base project](ttps://github.com/abhinav-upadhyay/chatgpt_plugins). 
