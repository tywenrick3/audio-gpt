import logging
import os
import openai
import soundfile as sf
from flask import Flask, render_template, request, flash, redirect, url_for
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from flask_uploads import UploadSet, configure_uploads, AUDIO
from wtforms import SubmitField
from werkzeug.utils import secure_filename
from config import Config
from pydub import AudioSegment
from pydub.silence import split_on_silence
import tempfile
import concurrent.futures
from ratelimiter import RateLimiter
import time
from requests.exceptions import RequestException
import uuid


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

flask_logger = logging.getLogger('werkzeug')
flask_logger.setLevel(logging.INFO)
app.config.from_object(Config)

# Configure uploaded audio file destination
if not os.path.exists('uploaded_audio'):
    os.makedirs('uploaded_audio')

openai.api_key = app.config['OPENAI_API_KEY']

custom_audio_extensions = AUDIO + ('m4', 'm4a')

audio_files = UploadSet('audio', custom_audio_extensions)

app.config['UPLOADED_AUDIO_DEST'] = 'uploaded_audio'
configure_uploads(app, audio_files)

rate_limiter = RateLimiter(max_calls=50, period=60)


class AudioForm(FlaskForm):
    file = FileField('Audio File', validators=[
        FileRequired(), FileAllowed(audio_files, 'Audio only!')])
    submit = SubmitField('Submit')


def transcribe_chunk(audio_chunk):
    MAX_RETRIES = 5
    INITIAL_DELAY = 1.0  # seconds

    # Create a temporary file to store the audio_chunk_data
    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio_file:
        audio_chunk.export(temp_audio_file.name, format="wav")
        temp_audio_file.flush()

        # retry logic attempt to avoid rate limit errors, redis task queing might be better solution
        retries = 0
        while retries < MAX_RETRIES:
            try:
                response = openai.Audio.transcribe(
                    "whisper-1", temp_audio_file)
                return response.text
            except RequestException as e:
                error_message = e.args[0].get("error", {}).get("message", "")
                if "Rate limit" in error_message:
                    wait_time = INITIAL_DELAY * (2 ** retries)
                    time.sleep(wait_time)
                    retries += 1
                else:
                    raise ValueError(f"Failed to transcribe audio: {e}")
            except Exception as e:
                raise ValueError(f"Failed to transcribe audio: {e}")

        # If all retries fail, raise an error
        raise ValueError(
            "Failed to transcribe audio: Rate limit reached after all retries.")


def transcribe_audio(file_path):
    start_time = time.time()
    try:
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_channels(1)  # Convert to mono
        sample_rate = audio.frame_rate
    except Exception as e:
        raise ValueError(f"Failed to read audio file: {e}")

    # Split the audio into chunks based on silence
    audio_chunks = split_on_silence(
        audio, min_silence_len=500, silence_thresh=-40)

    transcript = ""

    BATCH_SIZE = 20

    # changed from 4: TODO, record time diffrenece in transcription
    num_threads = 6
    # Use ThreadPoolExecutor to transcribe audio chunks concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        for i in range(0, len(audio_chunks), BATCH_SIZE):
            batch_chunks = audio_chunks[i:i + BATCH_SIZE]

            # Create a list of futures for the current batch
            futures = [executor.submit(transcribe_chunk, audio_chunk)
                       for audio_chunk in batch_chunks]

            # Wait for all futures to complete and collect results
            results = [future.result()
                       for future in concurrent.futures.as_completed(futures)]

            # Append the transcribed text to the transcript
            transcript += " ".join(results)

    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Transcription time: {time_taken:.2f} seconds")

    return transcript, time_taken


def generate_summary_and_title(transcript):
    """Generate a summary of an audio transcript using OpenAI GPT-4 API."""
    start_time = time.time()
    model = 'gpt-4'
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that accurately summarizes and titles audio transcripts. The title should be no longer than 10 words."},
                {"role": "user", "content": f"transcript: {transcript}"},
                {"role": "assistant", "content": "Please provide a title and a summary."},
            ],
            max_tokens=2000,
            n=1,
        )
    except Exception as e:
        raise ValueError(f"Failed to generate summary: {e}")

    summary = response['choices'][0]['message']['content']
    title, summary = summary.split('\n', 1)

    # Remove the first 7 characters, which is the length of "Title: "
    title = title[7:]
    # Remove the first 9 characters, which is the length of "Summary: "
    summary = summary[9:]

    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Summary generation time: {time_taken:.2f} seconds")

    return title.strip(), summary.strip(), time_taken


@app.route('/recorded-audio', methods=['POST'])
def recorded_audio():
    audio_data = request.files['file']
    unique_id = str(uuid.uuid4())
    filename = f"{unique_id}.wav"
    audio_path = os.path.join(app.config['UPLOADED_AUDIO_DEST'], filename)
    audio_data.save(audio_path)
    return {'id': unique_id}


@app.route('/summary/<unique_id>', methods=['GET'])
def recorded_audio_summary(unique_id):
    filename = f"{unique_id}.wav"
    audio_path = os.path.join(app.config['UPLOADED_AUDIO_DEST'], filename)

    try:
        logger.info("Transcribing recorded audio")
        transcript, transcription_time = transcribe_audio(audio_path)
        logger.info("Generating summary and title")
        title, summary, summary_time = generate_summary_and_title(transcript)
    except ValueError as e:
        logger.error(f"Error: {e}")
        flash(str(e), 'error')
        return redirect(url_for('index'))

    logger.info("Rendering summary")
    return render_template('summary.html', title=title, summary=summary, transcript=transcript, transcription_time=transcription_time, summary_time=summary_time)


@app.route('/', methods=['GET', 'POST'])
def index():
    form = AudioForm()
    if form.validate_on_submit():
        logger.info("File submitted")
        filename = secure_filename(form.file.data.filename)
        audio_path = os.path.join(app.config['UPLOADED_AUDIO_DEST'], filename)
        form.file.data.save(audio_path)
        logger.info(f"File saved to: {audio_path}")
        try:
            logger.info("Transcribing audio")
            transcript, transcription_time = transcribe_audio(audio_path)
            logger.info("Generating summary and title")
            title, summary, summary_time = generate_summary_and_title(
                transcript)
        except ValueError as e:
            logger.error(f"Error: {e}")
            flash(str(e), 'error')
            return redirect(url_for('index'))
        logger.info("Rendering summary")
        return render_template('summary.html', title=title, summary=summary, transcript=transcript, transcription_time=transcription_time, summary_time=summary_time)
    return render_template('index.html', form=form)


if __name__ == '__main__':
    app.run(debug=True)
