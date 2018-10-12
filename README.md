## The emotional speech generation project 

#### The code structure 

- main.py
- data_utils.py
- trainer.py 
- model_conv1d.py
- model_lstm.py
- model_conv2d.py
- model_wavenet.py

#### Data Process 

* Pyworld vocoder is used to extract feature 
* the spectrum is first normalized to the log scale, then further normalized to be within the range of [-1, 1], see in `data_utils.py`

#### Model

* [ ] conv 1d 
* [ ] conv 2d, max pooling over time 
* [ ] LSTM 
* [ ] WaveNet

#### Trainer 

#### Notes

* We pad/ truncate all data to the maximum length, which is 400 frames. 92% of the data is within this range
* The normalization of data may be discussed here. see the code for details