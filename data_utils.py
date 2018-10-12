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
import random
import pickle
import json
import time

FFT_SIZE = 1024
FS = 16000
F0_CEIL = 500
EPSILON = 1e-10

DEBUG = True

class Dataset(object):
  """The dataset class"""

  def __init__(self, config):
    self.file_path = config["file_path"]
    self.data_path = config["data_path"]
    self.max_len = config["max_len"]
    self.batch_size = config["batch_size"]

    self.speaker2id = None
    self.id2speaker = None
    self.emotion2id = None
    self.id2emotion = None
    self.speaker_f0_stats = None
    self.normalizer = None

    self.num_batches = {"train": 0, "test": 0}
    self.num_cases = {"train": 0, "test": 0}
    self.batch_ptr = {"train": 0, "test": 0}
    self.data = { "train": 
                    { "speakers": None, 
                      "emotions": None, 
                      "f0": None,  
                      "sp": None, 
                      "ap": None, 
                      "en": None,
                      "lens": None}, 
                  "test": 
                    { "speakers": None, 
                      "emotions": None, 
                      "f0": None,  
                      "sp": None, 
                      "ap": None, 
                      "en": None, 
                      "lens": None} }
    return 

  def build(self):
    """Build the dataset"""
    print("Building the dataset ... ")
    # load all the data
    data = read_data(self.file_path) # TBC
    (self.speaker2id, self.id2speaker, self.emotion2id, self.id2emotion, 
      self.speaker_f0_stats, self.normalizer) = statistics(data)

    random.shuffle(data)
    num_data = len(data)
    num_train = num_data / 9
    data_train = data[: num_train]
    data_test = data[num_train: ]

    # data list to numpy array
    (speakers_train, emotions_train, f0_train, sp_train, ap_train, en_train,
      lens_train) = build_array(data_train, self.speaker2id, self.emotion2id, 
      self.max_len, self.normalizer)
    self.save_data("train", speakers_train, emotions_train, f0_train, 
      sp_train, ap_train, en_train, lens_train)
    self.data["train"]["speakers"] = speakers_train
    self.data["train"]["emotions"] = emotions_train
    self.data["train"]["f0"] = f0_train
    self.data["train"]["sp"] = sp_train
    self.data["train"]["ap"] = ap_train
    self.data["train"]["en"] = en_train
    self.data["train"]["lens"] = lens_train

    (speakers_test, emotions_test, f0_test, sp_test, ap_test, en_test,
      lens_test) = build_array(data_test, self.speaker2id, self.emotion2id, 
      self.max_len, self.normalizer)
    self.save_data("test", speakers_test, emotions_test, f0_test, sp_test, 
      ap_test, en_test, lens_test)
    self.data["test"]["speakers"] = speakers_test
    self.data["test"]["emotions"] = emotions_test
    self.data["test"]["f0"] = f0_test
    self.data["test"]["sp"] = sp_test
    self.data["test"]["ap"] = ap_test
    self.data["test"]["en"] = en_test
    self.data["test"]["lens"] = lens_test

    self.save_pickle()

    # build batches
    self.num_cases["train"] = len(self.data["train"]["speakers"])
    self.num_batches["train"] = \
      self.num_cases["train"] / self.batch_size
    if(self.num_cases["train"] % self.batch_size != 0):
      self.num_batches["train"] += 1

    self.num_cases["test"] = len(self.data["test"]["speakers"])
    self.num_batches["test"] = \
      self.num_cases["test"] / self.batch_size
    if(self.num_cases["test"] % self.batch_size != 0):
      self.num_batches["test"] += 1

    print("%d batches in train, %d batches in test" % 
      (self.num_batches["train"], self.num_batches["test"]))
    return 


  def save_data(self, setname, speakers, emotions, f0, sp, ap, en, lens):
    """Save the numpy array dataset, the majority of the dataset"""
    print("saving %s dataset" % setname)
    data_path = self.data_path
    np.save(os.path.join(data_path, "speakers_" + setname), speakers)
    np.save(os.path.join(data_path, "emotions_" + setname), emotions)
    np.save(os.path.join(data_path, "f0_" + setname), f0)
    np.save(os.path.join(data_path, "sp_" + setname), sp)
    np.save(os.path.join(data_path, "ap_" + setname), ap)
    np.save(os.path.join(data_path, "en_" + setname), en)
    np.save(os.path.join(data_path, "lens_" + setname), lens)
    return
  
  def save_pickle(self):
    """Save the stuffs other than numpy array"""
    print("saving to pickle ... ")
    data_path = self.data_path
    obj_list = [self.speaker2id, self.id2speaker, self.emotion2id, 
      self.id2emotion, self.speaker_f0_stats, self.normalizer]
    fname_list = ["speaker2id.pkl", "id2speaker.pkl", "emotion2id.pkl", 
      "id2emotion.pkl", "speaker_f0_stats.pkl", "normalizer.pkl"]
    for obj, fname in zip(obj_list, fname_list):
      filepath = os.path.joion(data_path, fname)
      pickle.dump(obj, open(filepath, "wb"))
    return

  def load(self):
    """Load the dataset"""
    # load the numpy array
    print("Loading the dataset ...")
    data_path = self.data_path
    for setname in ["train", "test"]:
      speakers = np.load(
        os.path.join(data_path, "speakers_" + setname + ".npy"))
      self.data[setname]["speakers"] = speakers
      emotions = np.load(
        os.path.join(data_path, "emotions_" + setname + ".npy"))
      self.data[setname]["emotions"] = emotions
      f0 = np.load(
        os.path.join(data_path, "f0_" + setname + ".npy"))
      self.data[setname]["f0"] = f0
      sp = np.load(
        os.path.join(data_path, "sp_" + setname + ".npy"))
      self.data[setname]["sp"] = sp
      ap = np.load(
        os.path.join(data_path, "ap_" + setname + ".npy"))
      self.data[setname]["ap"] = ap
      en = np.load(
        os.path.join(data_path, "en_" + setname + ".npy"))
      self.data[setname]["en"] = en
      lens = np.load(
        os.path.join(data_path, "lens_" + setname + ".npy"))
      self.data[setname]["lens"] = speakers

    # load other pickle
    print("Loading the pickle ... s")
    data_path = self.data_path
    obj_list = [self.speaker2id, self.id2speaker, self.emotion2id, 
      self.id2emotion, self.speaker_f0_stats, self.normalizer]
    fname_list = ["speaker2id.pkl", "id2speaker.pkl", "emotion2id.pkl", 
      "id2emotion.pkl", "speaker_f0_stats.pkl", "normalizer.pkl"]
    for obj, fname in zip(obj_list, fname_list):
      filepath = os.path.joion(data_path, fname)
      obj = pickle.load(open(filepath, "rb"))

    # build batches
    self.num_batches["train"] = \
      len(self.data["train"]["speakers"]) / self.batch_size
    if(len(self.data["train"]["speakers"]) % self.batch_size != 0):
      self.num_batches["train"] += 1
    self.num_batches["test"] = \
      len(self.data["test"]["speakers"]) / self.batch_size
    if(len(self.data["test"]["speakers"]) % self.batch_size != 0):
      self.num_batches["test"] += 1
    return 

  def next_batch(self, setname):
    """get next batch, update the pointer"""
    ptr = self.batch_ptr[setname]
    batch_size = self.batch_size
    speakers = self.data[setname]["speakers"][ptr: ptr + batch_size]
    emotions = self.data[setname]["emotions"][ptr: ptr + batch_size]
    f0 = self.data[setname]["f0"][ptr: ptr + batch_size]
    sp = self.data[setname]["sp"][ptr: ptr + batch_size]
    ap = self.data[setname]["ap"][ptr: ptr + batch_size]
    en = self.data[setname]["en"][ptr: ptr + batch_size]
    lens = self.data[setname]["lens"][ptr: ptr + batch_size]
    self.batch_ptr[setname] += batch_size
    if(self.batch_ptr[setname] > self.num_cases[setname]): 
      self.batch_ptr[setname] = self.num_cases[setname] - batch_size
    return speakers, emotions, sp, lens

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
  print("reading raw file ...")
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
  print("%d file processed" % len(data))
  return data

def statistics(data):
  """Feature statistics

  NOTE: all statistics should be implemented in this function

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
  print("data statistics ... ")
  start_time = time.time()
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
  # plt.hist(length_stat, bins = 20)
  # plt.savefig("length_statistics.png")
  print("statistics time consumption: %.2f" % (time.time() - start_time))
  return (speaker2id, id2speaker, emotion2id, id2emotion, speaker_f0_stats,
    normalizer)

def build_array(data, speaker2id, emotion2id, max_len, normalizer):
  """transform the data to numpy array for quicker io"""
  print("building numpy array ... ")
  start_time = time.time()
  speakers = []
  emotions = []
  f0_all = []
  sp_all = []
  ap_all = []
  en_all = []
  lens_all = []

  for spk, emo, _, f0, sp, ap, en in data:
    speakers.append(speaker2id[spk])
    emotions.append(emotion2id[emo])
    lens = f0.shape[0]
    if(lens > max_len):
      f0 = f0[: max_len]
      sp = sp[: max_len]
      ap = ap[: max_len]
      en = en[: max_len]
      lens = max_len
    lens_all.append(lens)

    f0_ = np.zeros([max_len])
    f0_[: lens] = f0
    f0_all.append(f0_)

    sp_ = np.zeros([max_len, sp.shape[1]])
    sp_[: lens] = sp
    sp_all.append(sp_)

    ap_ = np.zeros([max_len, ap.shape[1]])
    ap_[: lens] = ap
    ap_all.append(ap_)

    en_ = np.zeros([max_len])
    en_[: lens] = en.flatten()
    en_all.append(en_)

  speakers = np.array(speakers)
  emotions = np.array(emotions)
  f0_all = np.array(f0_all)
  sp_all = np.array(sp_all)
  ap_all = np.array(ap_all)
  en_all = np.array(en_all)
  lens_all = np.array(lens_all)
  if(DEBUG):
    print("speakers shape: ", speakers.shape)
    print("emotions shape: ", emotions.shape)
    print("f0 shape: ", f0_all.shape)
    print("sp shape: ", sp_all.shape)
    print("ap shape: ", ap_all.shape)
    print("energy shape: ", en_all.shape)
    print("lens shape: ", lens_all.shape)
  print("array building time: %.2f" % (time.time() - start_time))
  return speakers, emotions, f0_all, sp_all, ap_all, en_all, lens_all

def test():
  """Test the data utility functions"""
  config = json.load(open("config.json"))
  data = read_data(config["file_path"])
  speaker2id, id2speaker, emotion2id, id2emotion, speaker_f0_stats,\
    normalizer = statistics(data)
  speakers, emotions, f0, sp, ap, en, lens = build_array(
    data, speaker2id, emotion2id, config["max_len"], normalizer)
  return