import dofile from require 'moonscript'
import thisfile from require 'paths'

dofile(thisfile 'BGRU.moon')
dofile(thisfile 'ContextTable.moon')
dofile(thisfile 'LookupTableW2V.moon')
dofile(thisfile 'Encoder.moon')
dofile(thisfile 'Decoder.moon')

init = (opts) ->
  if opts.decoding and opts.decoding ~= ''
    dofile(thisfile 'STDecoder.moon')
    return SkipThoughtsDecoder
  else
    dofile(thisfile 'ST.moon')
    return SkipThoughts

{ :init }
