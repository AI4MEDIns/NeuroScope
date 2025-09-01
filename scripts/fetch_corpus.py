from pathlib import Path
import shutil
RAW=Path('data/raw');RAW.mkdir(parents=True,exist_ok=True)
SEED=Path('data/samples/seed.jsonl')
OUT=RAW/'corpus.jsonl'
shutil.copy(SEED, OUT)
print('✓ 原始语料已就绪:', OUT)
