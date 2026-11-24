"""Write invented square images/masks, NOT MRI data or a medical benchmark."""
from pathlib import Path

import numpy as np
from PIL import Image

path = Path('synthetic-demo')
if path.exists():
    raise ValueError('synthetic-demo already exists; choose a clean working folder.')
path.mkdir()
rows = ['image,mask,patient']
for i in range(3):
    image = np.zeros((32, 32), dtype=np.uint8)
    image[8:24, 8:24] = 255
    Image.fromarray(image).save(path / f'image{i}.png')
    Image.fromarray(image).save(path / f'mask{i}.png')
    rows.append(f'image{i}.png,mask{i}.png,invented-patient-{i}')
(path / 'manifest.csv').write_text('\n'.join(rows) + '\n', encoding='utf-8')
print(path / 'manifest.csv')
