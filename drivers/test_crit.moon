require 'torch'
require 'nn'

import thisfile from require 'paths'
import dofile from require 'moonscript'

dofile(thisfile 'InvCrossEntropyCriterion.moon')
dofile(thisfile 'SIPCriterion.moon')

inf = math.huge
preds = torch.Tensor{
  {0.25, 0.25, 0.25, 0.25}
  {0.10, 0.20, 0.30, 0.40}
  {0.50, 0.50, 0.50, 0.50}
}
target = torch.Tensor{1, 0, 2}

expectedGradInput = torch.Tensor({
  {-0.75, 0.25, 0.25, 0.25}
  {0, 0, 0, 0}
  {0.25, -0.75, 0.25, 0.25}
})\div(2)

crit = with nn.MaskedCrossEntropyCriterion!
  \forward(preds, target)
  \backward(preds, target)

assert crit.mTarget\equal(torch.Tensor{1, 1, 2})
assert crit.output == math.log(4) -- 2 * 1/2 * log(4); size avg
assert crit.gradInput\equal(expectedGradInput)

SEQLEN = 3

seqPreds = preds\view(1, preds\size(1), preds\size(2))\expand(SEQLEN, preds\size(1), preds\size(2))
seqTargets = target\view(target\size(1), 1)\expand(target\size(1), SEQLEN)

seqCrit = with nn.SIPCriterion(nn.MaskedCrossEntropyCriterion!)
  \forward(seqPreds, seqTargets)
  \backward(seqPreds, seqTargets)

expectedSeqGradInput = expectedGradInput\view(1, expectedGradInput\size(1), expectedGradInput\size(2))\expandAs(seqPreds)

assert seqCrit.output == SEQLEN*math.log(4)
assert seqCrit.gradInput\equal(expectedSeqGradInput)
