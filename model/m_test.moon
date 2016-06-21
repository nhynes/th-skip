import dofile from require 'moonscript'
import thisfile from require 'paths'

dofile 'init.moon'

BSZ = 2
SEQLEN = 5
VSZ = 6

m = Model
  seqlen: SEQLEN,
  dim: 4,
  nRNNs: 1,
  vocabSize: VSZ

m\cuda!

sent = torch.rand(BSZ, 3)\mul(VSZ)\round!\cuda!
prevSent = torch.rand(BSZ, SEQLEN)\mul(VSZ)\round!\cuda!
nextSent = torch.rand(BSZ, SEQLEN)\mul(VSZ)\round!\cuda!

input = {sent, prevSent, nextSent}

m\forward(input)
print m.output[1]\size!

gradPrev = torch.rand(BSZ, SEQLEN, VSZ)\cuda!
gradNext = torch.rand(BSZ, SEQLEN, VSZ)\cuda!
gradOutput = {gradPrev, gradNext}

m\backward(input, gradOutput)
