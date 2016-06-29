SkipThoughtsDecoder, parent = torch.class('SkipThoughtsDecoder', 'nn.Container')

SkipThoughtsDecoder.__init = (opts) =>
  parent.__init(self)

  @dpnn_getParameters_found = true -- prevent dpnn's getParameters from searching table

  @encoder = torch.load(opts.decoding)

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
  sentlen = input\size(2) - 1
  @encoder\forward(input\narrow(2, 2, sentlen))   -- chop off EOS
  @decoder\forward({input\narrow(2, 1, sentlen), @encoder.output})
  @output

SkipThoughtsDecoder.updateGradInput = (input, gradOutput) =>
  sentlen = input\size(2) - 1
  @decoder\backward({input\narrow(2, 1, sentlen), @encoder.output}, gradOutput)
  @gradInput = @encoder\backward(input\narrow(2, 2, sentlen), @decoder.gradInput)
  @gradInput
