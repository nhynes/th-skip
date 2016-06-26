import dofile from require 'moonscript'
import thisfile from require 'paths'

dofile(thisfile 'BGRU.moon')
dofile(thisfile 'ContextTable.moon')
dofile(thisfile 'LookupTableW2V.moon')

init = (opts) ->
  dofile(thisfile 'Encoder.moon')

  Model = torch.class('Model', 'Encoder') -- temporary patch

  if opts.decoding and opts.decoding ~= ''
    dofile(thisfile 'Decoder.moon')
    return Decoder
  return Encoder

{ :init }
