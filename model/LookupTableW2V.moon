require 'torch'
require 'nn'
import dofile from require 'moonscript'
import thisfile from require 'paths'

loadW2V = dofile(thisfile 'w2v.moon')

LookupTableW2V, parent = torch.class('nn.LookupTableW2V', 'nn.LookupTable')

LookupTableW2V.__init = (nWords, nRandInit, w2vPath) =>
  w2v = loadW2V(w2vPath)
  @nOutput = w2v\size(2)

  nWordsW2V = math.min(nWords-nRandInit, w2v\size(1))
  parent.__init(self, nWordsW2V + nRandInit, @nOutput)

  @weight\sub(nRandInit+1, -1)\copy(w2v\sub(1, nWordsW2V))
