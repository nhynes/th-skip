require 'torch'

import dofile from require 'moonscript'
import thisfile from require 'paths'

dofile 'DataLoader.moon'

toks = torch.ByteTensor{
  {5, 5, 5, 0, 0} -- actual max length is 7

  {5, 5, 5, 5, 5}
  {7, 7, 7, 7, 7}

  {5, 5, 7, 6, 5}
  {6, 0, 0, 0, 0}

  {5, 6, 0, 0, 0}

  {5, 7, 6, 0, 0}
}

rlens = torch.LongTensor{1, 2, 2, 1, 1}
slens = torch.LongTensor{3, 5, 5, 5, 1, 2, 3}
rbps = torch.LongTensor{1, 2, 4, 6, 7}

ids = torch.CharTensor(7, 2)\zero!
ids[{i, 1}] = 48+i for i=1,7

data =
  ids: torch.rand(3, 10)\char!
  toks: toks
  rlens: rlens
  slens: slens
  rbps: rbps

opts =
  batchSize: 2

dl = DataLoader(data, data, opts)

print(dl\makebatch!)
