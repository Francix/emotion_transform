"""The emotinonal speech generation project 

Yao Fu, Columbia University
Yao.fu@columbia.edu
THU OCT 11TH 2018, 15:56
"""

from data_utils import Dataset
import json

def main():
  config = json.load(open("config.json"))
  dset = Dataset(config)
  return 

if __name__ == "__main__":
  main()