import math
from pathlib import Path
import ke

__all__ = [
    'get_file_size',
    'read_file',
]

logger = ke.logger

get_file_size = lambda file_path, decimal_places=2, binary=True: (lambda size_bytes: (lambda units, base: f"{size_bytes / (base ** (idx := min(math.floor(math.log(size_bytes, base)) if size_bytes else 0, len(units)-1))):.{decimal_places}f} {units[idx]}")(['B', 'KB', 'MB', 'GB', 'TB'], 1024 if binary else 1000))(Path(file_path).stat().st_size)

def read_file(file_path):
    logger.info(f'Load File from {file_path} ({get_file_size(file_path)})')
    with open(file_path, encoding='utf-8') as f:
        data, sentence, label = [], [], []
        for line in f:
            if len(line) == 0 or line[0] == '\n':
                if len(sentence) > 0:
                    data.append((sentence, label))
                    sentence, label = [], []
                continue
            splits = line.strip().split()
            sentence.append(splits[0])
            label.append(splits[-1])

    if len(sentence) > 0:
        data.append((sentence, label))
        sentence, label = [], []
    return data
