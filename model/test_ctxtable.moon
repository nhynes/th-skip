require 'torch'

import dofile from require 'moonscript'
import thisfile from require 'paths'

dofile(thisfile 'ContextTable.moon')

ct = nn.ContextTable!

base = torch.range(1, 2*3*4)\view(2, 3, 4) -- batch x seqlen x dim
ctx = torch.range(1, 2*4)\view(2, 4) -- batch x dim

expected = torch.Tensor{
  {
    {1, 2, 3, 4, 1, 2, 3, 4}
    {5, 6, 7, 8, 1, 2, 3, 4}
    {9, 10, 11, 12, 1, 2, 3, 4}
  }
  {
    {13, 14, 15, 16, 5, 6, 7, 8}
    {17, 18, 19, 20, 5, 6, 7, 8}
    {21, 22, 23, 24, 5, 6, 7, 8}
  }
}

actual = ct\forward{base, ctx}

assert actual\eq(expected)\sum! == 2*3*(4+4)
