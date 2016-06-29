import dofile from require 'moonscript'
import thisfile from require 'paths'

loadW2V = dofile(thisfile 'w2v.moon')

LookupTableW2V, parent = torch.class('nn.LookupTableW2V', 'nn.LookupTable')

LookupTableW2V.__init = (w2vPath, nWords, nRandInit=0) =>
  w2v = loadW2V(w2vPath)
  @nOutput = w2v\size(2)

  nWordsW2V = math.min(nWords-nRandInit, w2v\size(1))
  @nIndex = nWordsW2V + nRandInit
  parent.__init(self, @nIndex, @nOutput)

  @weight\sub(nRandInit+1, -1)\copy(w2v\sub(1, nWordsW2V))

LookupTableW2V.updateOutput = (input) =>
  parent.updateOutput(self, input)
  @output = @output\transpose(1, 2)
  @output

LookupTableW2V.updateGradInput = (input, gradOutput) =>
  parent.updateGradInput(self, input, gradOutput\transpose(1, 2))
  @gradInput

LookupTableW2V.accGradParameters = (input, gradOutput, scale) =>
  parent.accGradParameters(self, input, gradOutput\transpose(1, 2), scale)
