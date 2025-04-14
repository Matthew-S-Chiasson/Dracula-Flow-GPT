# Fine-tuning GPT models to rap like Dracula.

This project fine-tunes GPT-2 and GPT-Neo on the transcript data from the YouTube video series Dracula Flow 1–5. The goal is to generate new text in the same dark, absurd, and hilarious tone—aka Dracula Flow-style.

Project started in 2024, added to GitHub in 2025.

# Project Goal
- Use fine-tuning to teach large language models to mimic a niche, stylistic internet content format

- Explore differences in output between GPT-2 and GPT-Neo

- Learn more about dataset preparation, training loops, and prompt design for creative text generation

# Dataset
The dataset was created by manually transcribing and cleaning up the text from the Dracula Flow video series (1–5). This includes all verses, transitions, and comedic timing cues when relevant.

> Note: The transcripts are not included in this public repo for copyright reasons. However, they were sourced directly from publicly available YouTube content.

# Models Used
- GPT-2 Small (124M) from Hugging Face Transformers

- GPT-Neo 1.3B, also via Hugging Face

Both models were fine-tuned locally on a GTX 1080 Ti GPU using PyTorch and Hugging Face's Trainer API.

# How It Works
Load pre-trained GPT model

Prepare and tokenize Dracula Flow transcripts

Fine-tunes and saves checkpoints in the results folder. and saves final models in a floder named by the user.
> _Note:_ I couldnt push the full checkpoints i made as tehy were way to big and im to lazy to figure out a better way :)

Generate new "Dracula Flow" style text with temperature sampling

# Final Notes
This is a creative experiment that mixes AI with internet humor and music parody. Not intended for production—just a fun and weird little side quest into fine-tuning! Also the models will be capable of profamily.
