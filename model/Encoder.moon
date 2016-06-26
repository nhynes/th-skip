require 'torch'
require 'nn'
require 'cunn'
require 'nngraph'

Encoder, parent = torch.class('Encoder', 'nn.Container')

Encoder.__init = (opts) =>
  parent.__init(self)

  @lut = nn.LookupTableW2V(opts.vocabSize, 4, opts.w2v) -- UNK, <r>, </r>, <s>, </s>
  lutDecNext = @lut\clone('weight', 'gradWeight')()
  lutDecPrev = @lut\clone('weight', 'gradWeight')()
  wembDim = @lut.nOutput

  @encoder = with nn.Sequential!
    \add @lut
    \add cudnn.BGRU(wembDim, opts.dim, opts.nRNNs, true)
    \add nn.Select(2, -1)
    \add nn.Normalize(2)

  wordDec = nn.TemporalConvolution(opts.dim, opts.vocabSize, 1)

  @decNext = with nn.Sequential!
    \add nn.ContextTable(3)
    \add cudnn.GRU(2*opts.dim + wembDim, opts.dim, opts.nRNNs, true)
    \add wordDec
    \add nn.Transpose({1, 3})
    \add cudnn.SpatialLogSoftMax!

  @decPrev = with @decNext\clone!
    \applyToModules (mod) -> mod\reset!
    \get(3)\share(wordDec, 'weight', 'gradWeight', 'bias', 'gradBias')

  stVecs = self.encoder!
  prevPreds = self.decPrev{lutDecPrev, stVecs}
  nextPreds = self.decNext{lutDecNext, stVecs}

  @model = nn.gModule({stVecs, lutDecPrev, lutDecNext}, {prevPreds, nextPreds})

  @modules = {@model}

  collectgarbage!

Encoder.updateOutput = (input) =>
  [==[
  input: {sent, prevSent, nextSent} (N x sentlen, elems in [1, vocabSize])
  output: {predPrevSent, predNextSent} (sentlen x N x vocabSize)
  ]==]
  @output = @model\forward(input)
  @output

Encoder.updateGradInput = (input, gradOutput) =>
  @gradInput = @model\backward(input, gradOutput)
  @gradInput
