_ = require 'moses'

SkipThoughts, parent = torch.class('SkipThoughts', 'nn.Container')

SkipThoughts.__init = (opts) =>
  parent.__init(self)

  @encoder = Encoder(opts)

  decOpts = _.extend({wembDim: @encoder.lut.nOutput, embDim: @encoder.embDim}, opts)
  @decPrev = Decoder(decOpts, {lut: @encoder.lut})
  @decNext = Decoder(decOpts, {lut: @encoder.lut, dec: @decPrev.wordDec})

  @gradEncoder = torch.Tensor!

  @modules = {@encoder, @decPrev, @decNext}

  collectgarbage!

SkipThoughts.updateOutput = (input) =>
  [==[
  input: {sent, prevSent, nextSent} (N x sentlen, elems in [1, vocabSize])
  output: {predPrevSent, predNextSent} (sentlen x N x vocabSize)
  ]==]
  {sent, prevSent, nextSent} = input
  @encoder\forward(sent)
  @decPrev\forward({prevSent, @encoder.output})
  @decNext\forward({nextSent, @encoder.output})
  @output = {@decPrev.output, @decNext.output}
  @output

SkipThoughts.updateGradInput = (input, gradOutput) =>
  {sent, prevSent, nextSent} = input
  {gradPrev, gradNext} = gradOutput
  @decPrev\backward({prevSent, @encoder.output}, gradPrev)
  @decNext\backward({nextSent, @encoder.output}, gradNext)
  @gradEncoder\add(@decPrev.gradInput, 1, @decNext.gradInput)
  @gradInput = @encoder\backward(sent, @gradEncoder)
  @gradInput
