import sys
sys.path.append('../')

import numpy as np
import src.mylib.pgm
from src.mylib.pgm import create_node
from src.mylib.pgm import create_tensor
from src.mylib.pgm import create_node_from_pgm
from src.mylib.pgm import create_image_from_tensor
from src.mylib.pgm import crop_pgm_image

pgmf_path = "gray8bit_Dataset/"

def test_np_eye():
  I = np.eye(4)
  output = create_tensor(matrix=I, tensor_base=2, transpose_modes=False)
  assert output[1] == 4
    
def test_inversibility_no_transpose():
  I = np.random.rand(4, 4)
  tensor = create_tensor(matrix=I, tensor_base=2, transpose_modes=False)[0]
  output = create_image_from_tensor(tensor=tensor, transpose_modes=False)
  assert (output == I).all()

def test_inversibility_transpose():
  I = np.random.rand(4, 4)
  tensor = create_tensor(matrix=I, tensor_base=2, transpose_modes=True)[0]
  output = create_image_from_tensor(tensor=tensor, transpose_modes=True)
  assert (output == I).all()
    
def test_create_node():
  I = np.random.rand(4, 4)
  output = create_node(matrix=I, tensor_base=2, transpose_modes=False)
  assert output[1] == 4 and (I == create_image_from_tensor(tensor=output[0].tensor, transpose_modes=False)).all()