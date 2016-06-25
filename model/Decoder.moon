require 'torch'
require 'nn'
require 'cunn'
require 'nngraph'
import dofile from require 'moonscript'
import thisfile from require 'paths'

loadW2V = dofile(thisfile 'w2v.moon')
dofile(thisfile 'BGRU.moon')
dofile(thisfile 'ContextTable.moon')
dofile(thisfile 'DontTrain.moon')

loadEncoder = (snapPath) ->
  {:opts, :model} = torch.load(snapPath)
  model = model\get(1)
  enc = model\get(2)
  print(enc)
  enc, opts.dim*2

Model, parent = torch.class('Model', 'nn.Container')

Model.__init = (opts) =>
  parent.__init(self)

  encoder, stDim = loadEncoder(opts.decoding)

  decoder = with nn.Sequential!
    \add nn.ContextTable(3)
    \add cudnn.GRU(stDim, opts.dim, opts.nRNNs, true)
    \add nn.TemporalConvolution(opts.dim, opts.vocabSize, 1)
    \add nn.Transpose({1, 3})
    \add cudnn.SpatialLogSoftMax!

  stVecs = nn.DontTrain(encoder)()
  preds = decoder(stVecs)

  @model = nn.gModule({stVecs}, {preds})

  @modules = {@model, encoder, decoder}

  collectgarbage!

Model.updateOutput = (input) =>
  [==[
  input: sent (N x sentlen, elems in [1, vocabSize])
  output: preds (sentlen x N x vocabSize)
  ]==]
  @output = @model\forward(input)
  @output

Model.updateGradInput = (input, gradOutput) =>
  @gradInput = @model\backward(input, gradOutput)
  @gradInput
