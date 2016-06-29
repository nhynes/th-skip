require 'dpnn'

SkipThoughtsDecoder, parent = torch.class('SkipThoughtsDecoder', 'nn.Container')

SkipThoughtsDecoder.__init = (opts) =>
  parent.__init(self)

  @dpnn_getParameters_found = true -- prevent dpnn's getParameters from searching table

  @encoder = torch.load(opts.decoding).model\get(1)

  opts.wembDim = @encoder.lut.nOutput
  opts.embDim = @encoder.embDim
  @decoder = Decoder(opts, {lut: @encoder.lut})

  @modules = {@decoder}

  collectgarbage!

SkipThoughtsDecoder.updateOutput = (input) =>
  [==[
  input: {sent} (N x sentlen+1, elems in [1, vocabSize])
  output: predSent (sentlen x N x vocabSize)
  ]==]
  {encInp, decInp} = input
  @encoder\forward(encInp)
  @output = @decoder\forward({decInp, @encoder.output})
  @output

SkipThoughtsDecoder.updateGradInput = (input, gradOutput) =>
  {encInp, decInp} = input
  @decoder\backward({decInp, @encoder.output}, gradOutput)
  @gradInput = @encoder\backward(encInp, @decoder.gradInput[2])
  @gradInput
