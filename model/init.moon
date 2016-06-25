require 'torch'
require 'nn'
require 'cunn'
require 'nngraph'
import dofile from require 'moonscript'
import thisfile from require 'paths'

loadW2V = dofile(thisfile 'w2v.moon')
dofile(thisfile 'BGRU.moon')
dofile(thisfile 'ContextTable.moon')
dofile(thisfile 'LookupTableW2V.moon')

Model, parent = torch.class('Model', 'nn.Container')

Model.__init = (opts) =>
  parent.__init(self)

  lut = nn.LookupTableW2V(opts.vocabSize, 4, opts.w2v) -- UNK, <r>, </r>, <s>, </s>
  lutDecNext = lut\clone('weight', 'gradWeight')()
  lutDecPrev = lut\clone('weight', 'gradWeight')()
  wembDim = lut.nOutput

  encoder = with nn.Sequential!
    \add lut
    \add cudnn.BGRU(wembDim, opts.dim, opts.nRNNs, true)
    \add nn.Select(2, -1)
    \add nn.Normalize(2)

  wordDec = nn.TemporalConvolution(opts.dim, opts.vocabSize, 1)

  decNext = with nn.Sequential!
    \add nn.ContextTable(3)
    \add cudnn.GRU(2*opts.dim + wembDim, opts.dim, opts.nRNNs, true)
    \add wordDec
    \add nn.Transpose({1, 3})
    \add cudnn.SpatialLogSoftMax!

  decPrev = with decNext\clone!
    \applyToModules (mod) -> mod\reset!
    \get(4)\share(wordDec, 'weight', 'gradWeight', 'bias', 'gradBias')

  stVecs = encoder()
  prevPreds = decPrev{lutDecPrev, stVecs}
  nextPreds = decNext{lutDecNext, stVecs}

  @model = nn.gModule({stVecs, lutDecPrev, lutDecNext}, {prevPreds, nextPreds})

  @modules = {@model, encoder, decPrev, decNext, lutT}

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
