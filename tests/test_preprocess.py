from pathlib import Path

def test_samples_exists():
    assert Path('data/samples/seed.jsonl').exists()
