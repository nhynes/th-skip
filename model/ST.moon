_ = require 'moses'

SkipThoughts, parent = torch.class('SkipThoughts', 'nn.Container')

SkipThoughts.__init = (opts) =>
  parent.__init(self)
  @sameScale = opts.sameScale

  @encoder = Encoder(opts)

  decOpts = _.extend({wembDim: @encoder.lut.nOutput, embDim: @encoder.embDim}, opts)
  @decPrev = Decoder(decOpts, {lut: @encoder.lut})
  @decSame = Decoder(decOpts, {lut: @encoder.lut, dec: @decPrev.wordDec})
  @decNext = Decoder(decOpts, {lut: @encoder.lut, dec: @decPrev.wordDec})

  @gradEncoder = torch.Tensor!

  @modules = {@encoder, @decPrev, @decSame, @decNext}

  collectgarbage!

SkipThoughts.updateOutput = (input) =>
  [==[
  input: {sent, prevSent, nextSent} (N x sentlen, elems in [1, vocabSize])
  output: {predPrevSent, predNextSent} (sentlen x N x vocabSize)
  ]==]
  {sent, prevSent, nextSent} = input
  sentlen = sent\size(2)-1

  @encoder\forward(sent\narrow(2, 2, sentlen))

  @decPrev\forward({prevSent, @encoder.output})
  @decSame\forward({sent\narrow(2, 1, sentlen), @encoder.output})
  @decNext\forward({nextSent, @encoder.output})

  @output = {@decPrev.output, @decSame.output, @decNext.output}
  @output

SkipThoughts.updateGradInput = (input, gradOutput) =>
  {sent, prevSent, nextSent} = input
  {gradPrev, gradSame, gradNext} = gradOutput
  sentlen = sent\size(2)-1

  @decPrev\backward({prevSent, @encoder.output}, gradPrev)
  @decSame\backward({sent\narrow(2, 1, sentlen), @encoder.output}, gradSame)
  @decNext\backward({nextSent, @encoder.output}, gradNext)

  @gradEncoder\add(@decPrev.gradInput[2], @decNext.gradInput[2])
  @gradEncoder\add(@decSame.gradInput[2], @sameScale)
  @gradInput = @encoder\backward(sent\narrow(2, 2, sentlen), @gradEncoder)
  @gradInput
