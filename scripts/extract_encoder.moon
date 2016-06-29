require 'dpnn'

import dofile from require 'moonscript'
import thisfile from require 'paths'

dofile(thisfile '../model/init.moon').init{}

{snap, outfile, eval} = arg

{:model, :opts} = torch.load(snap)

encoder = with nn.Serial(model\get(1).encoder)
  \evaluate! if eval
  \lightSerial! if eval else \mediumSerial!

torch.save(outfile, {model: encoder, :opts})
