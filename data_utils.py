"""The emotinonal speech generation project, data processing functions

Yao Fu, Columbia University
Yao.fu@columbia.edu
THU OCT 11TH 2018, 15:56
"""

import numpy as np 
import librosa 
import pyworld as pw 
import os
import matplotlib.pyplot as plt
import tqdm

FFT_SIZE = 1024
FS = 16000
F0_CEIL = 500
EPSILON = 1e-10

class Dataset(object):
  """The dataset class"""

  def __init__(self, config):
    self.file_path = config["file_path"]
    return 

  def build(self):
    """Build the dataset"""
    ## load all the data
    data = read_data(self.file_path) # TBC
    (speaker2id, id2speaker, emotion2id, id2emotion, speaker_f0_stats,
      normalizer) = statistics(data)
    
    return 


  def save(self):
    """Save the dataset"""
    return

  def load(self):
    """Load the dataset"""
    return 

  def next_batch(self):
    return

class Normalizer(object):
  """Normalize the spectrum to [-1, 1] """
  def __init__(self, xmin, xmax):
    self.xmin = xmin
    self.xmax = xmax
    self.xscale = xmax - xmin
    return
  
  def forward_process(self, x):
    """Perform nornalization"""
    assert(type(x) == np.ndarray)
    x = (x - self.xmin) / self.xscale
    x = x.clip(min = 0.0, max = 1.0) * 2.0 - 1
    return x

  def backward_process(self, x):
    """De-normalizaion"""
    return (x * .5 + .5) * self.xscale + self.xmin


def wavfile2pw(filename, f0_ceil = F0_CEIL, fs = FS, fft_size = FFT_SIZE):
  """Speech analysis given the file name
  
  We use the PyWorld to extract feature, following the practice in:
  https://github.com/JeremyCCHsu/vae-npvc

  NOTE: The spectrum is normalized by energy and transformed to log scale. 
  To be discussed here 

  After transforming to the log scale, the spectrum will be further 
  normalized to be in the range of [-1, 1]
  
  Args:
    filename: the wav file 
    f0_ceil: maximum f0, note here we set the default to be 500, while praat 
      suggest we set 250. this will result in many small values in high frequence, probably not learnable for a network
    fs: sampling frequency, librosa will handle the frequency conversion
      from the original wavfile 
    fft_size: fft size

  Returns:
    f0: the pitch/ fundamental frequencys
    sp: spectogram
    ap: aperiodicity
    en: energy
  """
  x, _ = librosa.load(filename, sr = fs, mono = True, dtype = np.float64)
  _f0, t = pw.dio(x, fs, f0_ceil = f0_ceil)
  f0 = pw.stonemask(x, _f0, t, fs)
  sp = pw.cheaptrick(x, f0, t, fs, fft_size = fft_size)
  ap = pw.d4c(x, f0, t, fs, fft_size = fft_size)
  en = np.sum(sp + EPSILON, axis=1, keepdims=True)
  sp = np.log10(sp / en)
  return f0, sp, ap, en

def pw2wavfile():
  """speech synthesis given the feature"""
  # TBC
  return

def read_data(file_path):
  """Read the raw data, extract feature"""
  wav_files = [f for f in os.listdir(file_path) if f.endswith(".wav")]
  data = []
  for wf in tqdm.tqdm(wav_files):
    # get the speaker, the content, and the emotion
    wf_ = wf.split("_")
    speaker = wf_[0]
    emotion = wf_[2]
    content = wf_[4].split(".")[0]
    f0, sp, ap, en = wavfile2pw(os.path.join(file_path, wf))
    data.append([speaker, emotion, content, f0, sp, ap, en])
  print("%d data processed" % len(data))
  return data

def statistics(data):
  """Feature statistics

  Args:
    data: the dataset as a list of 
      [speaker, emotion, content, f0, sp, ap, en] pairs

  Returns:
    speaker2id: as the name suggests
    id2speaker: as the name suggests 
    emotion2id: as the name suggests
    id2emotion: as the name suggests 
    speaker_f0_stats: the mean and deviation of each speaker's f0
    normalizer: the spectrum normalizer 
  """
  speaker2id = dict()
  id2speaker = dict()
  num_speaker = 0
  emotion2id = dict()
  id2emotion = dict()
  num_emotion = 0
  length_stat = []
  sp_all = []
  speaker_f0 = dict()
  speaker_f0_stats = dict()

  for spk, emo, ctt, f0, sp, _, _ in data:
    if(spk not in speaker2id):
      speaker2id[spk] = num_speaker
      id2speaker[num_speaker] = spk
      num_speaker += 1
    if(emo not in emotion2id):
      emotion2id[emo] = num_emotion
      id2emotion[num_emotion] = emo
      num_emotion += 1
    slen = len(sp)
    length_stat.append(slen)
    sp_all.append(sp)

  # f0 amoung different speakers
  for spk in speaker2id: 
    speaker_f0[spk] = []
    speaker_f0_stats[spk] = {"mu": 0.0, "std": 0.0}
  for spk, emo, ctt, f0, sp, _, _ in data:
    speaker_f0[spk].append(f0)
  for spk in speaker_f0:
    spk_f0 = np.concatenate(speaker_f0[spk], axis=0)
    spk_f0 = spk_f0[spk_f0 > 2.0]
    spk_f0 = np.log(spk_f0)
    mu, std = spk_f0.mean(), spk_f0.std()
    speaker_f0_stats[spk]["mu"] = mu
    speaker_f0_stats[spk]["std"] = std
    print("speaker %s, f0 mu: %.4f, std: %.4f" % 
      (spk, speaker_f0_stats[spk]["mu"], speaker_f0_stats[spk]["std"]))
  
  # spectrum percentile and normalizer
  sp_all = np.concatenate(sp_all, axis=0)
  print("spectrum shape: ", sp_all.shape)
  sp_005 = np.percentile(sp_all, 0.5, axis=0)
  sp_995 = np.percentile(sp_all, 99.5, axis=0)
  normalizer = Normalizer(sp_005, sp_995)
  print("sp threshold shape", sp_005.shape)
  
  # length distribution, we will use it to determine the maximum length
  plt.hist(length_stat, bins = 20)
  plt.savefig("length_statistics.png")
  return (speaker2id, id2speaker, emotion2id, id2emotion, speaker_f0_stats,
    normalizer)

def test():
  """Test the data utility functions"""
  file_path = \
  "/Users/Francis_Yao/Documents/Columbia/EmotionalSpeech/EPSaT/wav"
  data = read_data(file_path)
  (speaker2id, id2speaker, emotion2id, id2emotion, speaker_f0_stats,
    normalizer) = statistics(data)
  return