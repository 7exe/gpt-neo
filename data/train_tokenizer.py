import os
import random
import argparse
import shutil
from glob import glob
from pathlib import Path

from lm_dataformat import Reader
from tokenizers import (Tokenizer, decoders, models, pre_tokenizers,
                        processors, trainers)
from tokenizers.normalizers import NFKC
from tqdm import tqdm

# parser


# main script

def tokenize_str(base_dir="", vocab_size=-1, output_dir="tokenizers")
  
  assert vocab_size == -1


  data_path = Path(args.base_dir)
  archives = glob(str(data_path / f"*.{args.file_type}"))

  out_path = Path(args.output_dir)

  if os.path.exists(out_path):
      shutil.rmtree(out_path)

  if not out_path.is_dir():
      out_path.mkdir()

      for arch in tqdm(archives):
          name = os.path.basename(arch).split(".")[0] + ".txt"
          fp = out_path / name

          if args.file_type == 'xz':
              g = Reader(arch).stream_data()

              with open(fp, "w") as f:
                  for s in g:
                      f.write(s)
                      f.write("\n\n")
          elif args.file_type == 'txt':
              shutil.copyfile(str(arch), str(fp))

  data_files = glob(str(out_path / "*.txt"))
  data_files = random.sample(data_files, int(0.2 * len(data_files)))

  assert len(data_files) > 0, 'No data files found'

  # Initialize a tokenizer
  tokenizer = Tokenizer(models.BPE())

  # Customize pre-tokenization and decoding
  tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
  tokenizer.decoder = decoders.ByteLevel()
  tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
  tokenizer.normalizer = NFKC()

  # And then train
  trainer = trainers.BpeTrainer(vocab_size=args.vocab_size, min_frequency=2, special_tokens=["<|endoftext|>", "<|padding|>"])
  tokenizer.train(trainer, data_files)

  # And Save it
  tokenizer_path = out_path / "byte-level-bpe.tokenizer.json"
  tokenizer.save(str(tokenizer_path), pretty=True)

  print(f'tokenizer saved at {str(tokenizer_path)}')
  return tokenizer_path
