# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # init

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### install libs

# COMMAND ----------

# MAGIC %pip install python-magic
# MAGIC %pip install ffprobe
# MAGIC %pip install pydub

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### imports

# COMMAND ----------

import os
import io
import magic
from pydub import AudioSegment

import pyspark.sql.functions as F
import pyspark.sql.types as T

from synapse.ml.cognitive import SpeechToTextSDK

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### speech to text SDK

# COMMAND ----------

# Run the Speech-to-text service to translate the audio into text
speech_to_text = (
    SpeechToTextSDK()
    .setSubscriptionKey("120889687b2b41eebcbf0e6f1ca9000a") # I know this is bad but it's only for exploration/demo purposes
    .setLocation("westeurope")
    .setOutputCol("text")
    .setAudioDataCol("audio_data")
    .setLanguage("cs-CZ")
    .setFileType("wav")
)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### paths & other

# COMMAND ----------

_DBFS_HOME_DIR = "/dbfs/FileStore/cdc_recordings/"
_TEST_DIR = f"{_DBFS_HOME_DIR}test/"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Data preprocessing and loading

# COMMAND ----------

 def read_and_preprocess_audio(file):        
    # read audio file, preprocess to meet speech-to-text API requirements
    # and return bytes stream 
    # requirements = .wav, 16000 frame-rate, mono/single-channel audio     
    
    mime_magic = magic.Magic(mime=True)    
    mime_type = mime_magic.from_file(file)
    
    if mime_type == "audio/x-wav":
        sound = AudioSegment.from_wav(file)
    elif mime_type == "audio/mpeg":
        sound = AudioSegment.from_mp3(file)
    elif mime_type in ["audio/ogg", "application/ogg"]:
        sound = AudioSegment.from_ogg(file)
    else:
        return b""
                   
    sound = sound.set_channels(1)
    sound = sound.set_frame_rate(16000)        

    sound_bytesio = io.BytesIO()
    sound.export(sound_bytesio, format='wav')
    preprocessed_sound = sound_bytesio.read()
            
    return preprocessed_sound
    
        
def get_dbfs_single_file(file_path: str):
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return [file_path]
    else:        
        print(f"ERROR: file not found: {file_path}")
        return []
    
    
def get_dbfs_folder(dir_path: str):
    files = []
    
    for root, subs, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            files.append(file_path)
    
    if not files:
        print(f"ERROR: no files found in: {dir_path}")
    
    return files
    
    
def load_audio_files(file_paths):                
    get_bytes = F.udf(read_and_preprocess_audio, T.BinaryType())
    
    df = (spark
        .createDataFrame(
            data=file_paths, 
            schema=T.StringType()
        )
        .withColumnRenamed('value', 'file')    
        .withColumn('audio_data', get_bytes("file"))
    )
    
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Transcription

# COMMAND ----------

def transcribe(speech_to_text, df):
    assert "audio_data" in df.columns
    
    return (
        speech_to_text
        .transform(df)        
        .withColumn("transcription", F.col("text.DisplayText"))
        .withColumn("tr_status", F.col("text.RecognitionStatus"))
        .drop("text")
    )


def transcribe_dbfs_single_file(speech_to_text, file_path):
    file_paths = get_dbfs_single_file(file_path)    
    df = load_audio_files(file_paths)    
    return transcribe(speech_to_text, df)


def transcribe_dbfs_dir(speech_to_text, dir_path):
    file_paths = get_dbfs_folder(dir_path)
    df = load_audio_files(file_paths)
    return transcribe(speech_to_text, df)

# COMMAND ----------

df = transcribe_dbfs_single_file(speech_to_text, f"{_TEST_DIR}voxpopuli_sample1.wav")
df.display()

# COMMAND ----------


