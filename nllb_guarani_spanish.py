# Translate from Guarani to Spanish
# Fine tune nllb model in the same way as the tyvan russian tutorial

import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_constant_schedule_with_warmup, NllbTokenizer
import re
from tqdm.auto import tqdm, trange
import random
import sys
import unicodedata
from sacremoses import MosesPunctNormalizer
from transformers.optimization import Adafactor
import gc 
import torch
import sacrebleu 
import numpy as np
from datasets import load_dataset
from collections import Counter
