require 'torch'
require 'cutorch'
require 'rnn'

import dofile from require 'moonscript'
import thisfile from require 'paths'

dofile(thisfile 'LMCriterion.moon')

init = (model, workers, opts) ->
  crit = with nn.ParallelCriterion!
    \add nn.SequencerCriterion(nn.LMCriterion!)
    \add nn.SequencerCriterion(nn.LMCriterion!)

  state =
    gpuSents: torch.CudaTensor(opts.batchSize, opts.sentlen)
    gpuNextSents: torch.CudaTensor(opts.batchSize, opts.sentlen)
    gpuPrevSents: torch.CudaTensor(opts.batchSize, opts.sentlen)
    crit: crit\cuda!
    t: 0

  drivers = {}
  lazyDrivers = {}

  for i, driver in pairs {'train', 'val', 'snap'}
    drivers[i] = (...) -> lazyDrivers[i](...)
    lazyDrivers[i] = (...) ->
      lazyDrivers[i] = dofile(thisfile driver..'.moon')(model, workers, opts, state)
      lazyDrivers[i](...)

  table.unpack drivers

{ :init }
