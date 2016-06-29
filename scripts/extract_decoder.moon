require 'dpnn'

import dofile from require 'moonscript'
import thisfile from require 'paths'

dofile(thisfile '../model/init.moon').init{decoding: true}

{snap, outfile, eval} = arg

{:model, :opts} = torch.load(snap)

decoder = with nn.Serial(model\get(1).decoder)
  \evaluate! if eval
  \lightSerial! if eval else \mediumSerial!

torch.save(outfile, {model: decoder, :opts})
