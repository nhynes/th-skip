require 'cunn'
require 'cudnn'

import dofile from require 'moonscript'
import thisfile from require 'paths'

dofile(thisfile 'ContextTable.moon')
dofile(thisfile 'LookupTableW2V.moon')
dofile(thisfile 'Din.moon')
dofile(thisfile 'Encoder.moon')
dofile(thisfile 'Decoder.moon')

init = (opts) ->
  if opts.decoding and opts.decoding ~= ''
    require 'dpnn'
    dofile(thisfile 'STDecoder.moon')
    return SkipThoughtsDecoder
  else
    dofile(thisfile 'ST.moon')
    return SkipThoughts

nn.Module.dontTrain = =>
  @parameters = =>
  @accGradParameters = =>
  @dpnn_getParameters_found = true
  self

nn.Container.dontTrain = =>
  @applyToModules (mod) -> mod\dontTrain!
  nn.Module.dontTrain(self)

{ :init }
