require 'torch'
require 'nn'
require 'cunn'
require 'nngraph'

loadEncoder = (snapPath) ->
  {:opts, :model} = torch.load(snapPath)
  model = model\get(1)
  enc = model\get(2)
  lut = enc\get(1)
  enc, lut, opts.dim*2

Decoder, parent = torch.class('Decoder', 'nn.Container')

Decoder.__init = (opts) =>
  parent.__init(self)

  @encoder, encLut, stDim = loadEncoder(opts.decoding)

  @lut = encLut\clone!

  @decoder = with nn.Sequential!
    \add nn.ContextTable()
    \add cudnn.GRU(stDim + @lut.nOutput, opts.dim, opts.nRNNs, true)
    \add nn.TemporalConvolution(opts.dim, opts.vocabSize, 1)
    \add nn.Transpose({1, 3})
    \add cudnn.SpatialLogSoftMax!

  wembs = self.lut!
  stVecs = self.encoder!
  preds = self.decoder{wembs, stVecs}

  @model = nn.gModule({stVecs, wembs}, {preds})

  @modules = {@decoder}

  collectgarbage!

Decoder.updateOutput = (input) =>
  [==[
  input: sent (N x sentlen, elems in [1, vocabSize])
  output: preds (sentlen x N x vocabSize)
  ]==]
  -- assert @model\parameters![1]\storage! ~= @decoder\parameters![1]\storage!
  @output = @model\forward(input)
  @output

Decoder.updateGradInput = (input, gradOutput) =>
  @gradInput = @model\backward(input, gradOutput)
  @gradInput
