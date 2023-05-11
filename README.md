# Audio Transcription and Summary Generator

This is a Flask web application that allows users to upload or record audio files, transcribes the audio using OpenAI's Whisper ASR, and generates a summary and title using OpenAI's GPT-4 API.

## Table of Contents

-   [Installation](#installation)
-   [Usage](#usage)
-   [Features](#features)

## Installation

1. Clone the repository

`git clone https://github.com/yourusername/your-repository.git`

2. Change directory into the repository

`cd your-repository`

3. Install required packages:

`pip install -r requirements.txt`

4. Create a `config.py` file in the root directory of the project and add the following content:

`OPENAI_API_KEY=your_openai_api_key`

## Usage

To run the web application locally, simply execute the following command in the terminal:

`python3 app.py`

Visit `http://127.0.0.1:5000/` in your web browser to access the application.

## Features

-   Upload audio files in various formats (WAV, MP3, MP4, etc.)
-   Record audio directly from the browser
-   Transcribe audio using OpenAI's Whisper ASR
-   Generate a summary and title using OpenAI's GPT-4 API

## Screenshots
<img width="1504" alt="Screenshot 2023-05-09 at 9 02 48 PM" src="https://github.com/tywenrick3/audio-gpt/assets/60354054/778741c5-a100-450e-9015-052d360afab3">

<img width="1504" alt="Screenshot 2023-05-09 at 9 02 56 PM" src="https://github.com/tywenrick3/audio-gpt/assets/60354054/f96a7660-ef82-4c84-a99d-bec559c4bf30">

