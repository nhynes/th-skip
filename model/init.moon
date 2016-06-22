require 'torch'
require 'nn'
require 'cunn'
require 'nngraph'
import dofile from require 'moonscript'
import thisfile from require 'paths'

loadW2V = dofile(thisfile 'w2v.moon')
dofile(thisfile 'BGRU.moon')
dofile(thisfile 'ContextTable.moon')

Model, parent = torch.class('Model', 'nn.Container')

Model.__init = (opts) =>
  parent.__init(self)

  -- initialize with w2v embeddings
  w2v = loadW2V(opts.w2v)
  nWords = math.min(opts.vocabSize, w2v\size(1))
  wembDim = w2v\size(2)
  w2v = w2v\sub(1, nWords)

  lut = nn.LookupTable(nWords+1, wembDim) -- + 1 for unk
  lut.weight\narrow(1, 2, nWords)\copy(w2v)

  lutT = with nn.Sequential!
    \add lut
    \add nn.Transpose({1, 2})

  lutDecNext = lutT\clone('weight', 'gradWeight')()
  lutDecPrev = lutT\clone('weight', 'gradWeight')()
  lutEnc = lutT()

  encoder = with nn.Sequential!
    \add cudnn.BGRU(w2v\size(2), opts.dim, opts.nRNNs)
    \add nn.Select(1, -1)

  decNext = with nn.Sequential!
    \add nn.ContextTable(3)
    \add cudnn.GRU(2*opts.dim + wembDim, opts.dim, opts.nRNNs)
    \add nn.Narrow(1, 1, -2) -- output after trailing </s> is forwarded is junk
    \add nn.TemporalConvolution(opts.dim, opts.vocabSize, 1)
    \add nn.Transpose({1, 3})
    \add cudnn.SpatialLogSoftMax!
    \add nn.Transpose({3, 1})

  decPrev = with decNext\clone!
    \applyToModules (mod) -> mod\reset!

  stVecs = encoder(lutEnc)
  prevPreds = decPrev{lutDecPrev, stVecs}
  nextPreds = decNext{lutDecNext, stVecs}

  @model = nn.gModule({lutEnc, lutDecPrev, lutDecNext}, {prevPreds, nextPreds})

  @modules = {@model, encoder, decPrev, decNext}

  collectgarbage!

Model.updateOutput = (input) =>
  [==[
  input: {sent, prevSent, nextSent} (N x sentlen, elems in [1, vocabSize])
  output: {predPrevSent, predNextSent} (sentlen x N x vocabSize)
  ]==]
  @output = @model\forward input
  @output

Model.updateGradInput = (input, gradOutput) =>
  @gradInput = @model\backward input, gradOutput
  @gradInput
