require 'torch'
require 'cutorch'
require 'cunn'
require 'rnn'

import dofile from require 'moonscript'
import thisfile from require 'paths'

dofile(thisfile 'LMCriterion.moon')

crit = nn.SequencerCriterion(nn.LMCriterion!)\cuda!

targets = torch.CudaTensor{ -- sent1    sent2    sent3
  {1, 1, 1},                -- blah     blah     blah
  {1, 2, 1},                -- blah     </s>     blah
  {2, 0, 1},                -- </s>              blah
  {0, 0, 2},                --                   </s>
  {0, 0, 0}                 -- disregard the output from forwarding the final </s>
}
wordProbs = torch.CudaTensor{ -- maxseqlen, batchsize, wordprobs (nll probs)
  {{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}},
  {{0.0, 1.0}, {1.0, 0.0}, {0.0, 1.0}},
  {{1.0, 0.0}, {0.0, 0.0}, {0.0, 1.0}},
  {{0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}},
  {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
}

print crit\forward(wordProbs, targets)
