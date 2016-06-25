import dofile from require 'moonscript'
import thisfile from require 'paths'

init = (opts) ->
  dofile(thisfile (opts.decoding ~= '' and 'Decoder' or 'Encoder')..'.moon')
  Model

{ :init }
