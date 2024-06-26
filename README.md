# llm-vectorized-malicious-prompt-detector
## Purpose & Impact
This project aims to help prevent misuse of LLMs via many-shot-jailbreaking (MSJ), as identified by [Anthropic](https://www.anthropic.com/research/many-shot-jailbreaking) (see [full research paper here](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://www-cdn.anthropic.com/af5633c94ed2beb282f6a53c595eb437e8e7b630/Many_Shot_Jailbreaking__2024_04_02_0936.pdf)). Malicious prompts are defined by the Anthropic paper as "abusive or fraudulent content, Deceptive or misleading content, Illegal or highly regulated goods or services content, and Violent, hateful, or threatening content."

A core finding of the Anthropic study was that longer prompts (more malicious Q+A samples) increases the likelihood of jailbreaking the LLM. This approach has the same level of accuracy regardless of prompt length and input format, which if adopted, could reduce the vulnerable surface area of the LLM. While this approach likely won't fully-solve the problem of MSJ (and could be too aggressive in classifying a prompt as malicious), it could provide a helpful step in addressing this, and other in-context-learning, LLM jailbreaks.

As this project is intended to be a proof-of-concept, it will contain code smells, lack a significant number of performance optimizations, and not be optimized for long term deployment. 

## Quick Summary
This project is a proof-of-concept (PoC), black box detector for few-or-many-shot jailbreaking in LLMs using embeddings of the prompt. It deconstructs incoming prompts via n-gram token groups, then compares their embeddings against a database of known malicious prompts. If similarity exceeds a set threshold, the prompt is blocked, enhancing LLM security.

## Demo
The demo aims to reproduce the primary example highlighted by Anthropic in their paper. The demo first shows ChatGPT being tricked into providing some information that would be useful to a nefarious actor; then, once the Detector is applied via the toggle button, shows that the prompt is flagged as malicious.

A `MALICIOUS_PROMPT_SIMILARITY_THRESHOLD` of 0.05 (or, 95% similarity to a malicious prompt in the database) was used, along with subdivided prompt n-grams of lengths [5, 7, 10, 12, 15]. The test was performed on GPT-3.5-Turbo (via Chat Completions API) using `text-embedding-ada-002`.

![MSJ Prompt Demo](demo.gif)

## Technical Details
The project uses Flask and a simple html front-end. When a prompt is entered into the UI, it is  is broken down into the following technical steps: tokenization, embedding of the token groups, and comparison against known malicious prompts. 

### Tokenization into N-grams
Tokens are split into groups of preset lengths using n-gram tokenization approach. The methodology and function used for this is [advertools' word_tokenizer](https://advertools.readthedocs.io/en/master/advertools.word_tokenize.html). Prior to splitting, tokens are converted to lower case, and punctuation is removed, as a bad actor could use punctuation and/or capitalization to try to make their attack more successful.

### Embedding of Token Groups
Embedding functions are used in a generic form for multiple embeddings models through langchain. Each group of tokens is embedded separately.

### Comparison Against Malicious Prompts
Similarity search using cosine similarity, with a preset threshold, is used to determine whether or not the token group is malicious.

### Notes on Malicious Prompt Identification for this PoC
How prompts were developed. 
1. I began with [learnprompting.org's Jailbreaking article](https://learnprompting.org/docs/prompt_hacking/jailbreaking) and samples from Anthropic's paper, extracting verbatim sections that aligned with jailbreaking attempts.
2. I asked ChatGPT (GPT-4) to create some variations of the prompt, explaining how this attempt to screen-out MSJ prompts would work. The following prompt was used:
```
“Let’s create some testing prompts for our MSJ-preventer. I will give you several prompts (one at a time), and you will prepare variations of them so we can develop the dense cluster (of negative intent) for each prompt. Please consider different subjects, objects, etc. that a malicious actor could use to phrase the same malicious intent. “
```

## Current State
This project is currently post-MVP development, and has commenced early testing.

# Running the application (Tuned for Windows for the PoC)
1. Install the dependencies: 
```
pip install openai --upgrade
pip install flask requests python-dotenv
```

2. Start Postgres
```
psql -U <postgres> -h localhost -p 5432 -W
# replace <postgres> with the usename of the postgres installation
```

3. Ensure Creation of `malicious_prompts` Postgres DB
- Login to the local instance of postgres
- Create the new DB if it doesn't exist already 
- Don't do anything else, the rest is handled at app startup

4. Start the app
```
python -m main
```

# Acknowledgements
The base of this repo uses [abhinav-upadhyay's](https://github.com/abhinav-upadhyay) project, [chatgpt_plugins](https://github.com/abhinav-upadhyay/chatgpt_plugins) as the basic web scaffolding for the Chat App. 
