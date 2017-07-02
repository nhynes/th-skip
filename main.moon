import dofile from require 'moonscript'

require 'torch'
require 'cutorch'
require 'drivers.CallbackQueue'

args = require 'args'
model = require 'model.init'
loader = require 'loader.init'
drivers = require 'drivers.init'

opts = args.parse arg

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opts.seed)
math.randomseed(opts.seed)

cutorch.setDevice(opts.gpu+1)

Model = model.init(opts)
theModel = nil
if paths.filep opts.loadsnap
  print 'Loading model from '..opts.loadsnap
  require 'dpnn'
  _ = require 'moses'
  snap = torch.load(opts.loadsnap)
  theModel = snap.model\get(1)
  newOpts = _.pick(opts, 'loadsnap', 'niters', 'dispfreq', 'valfreq', 'savefreq')
  opts = _.extend(snap.opts, newOpts)
  opts.savedState = snap.state
  print 'Resuming training from iteration '..opts.savedState.t
else
  theModel = Model(opts)
theModel\cuda!

workers = loader.init(opts)

train, val, snap = drivers.init(theModel, workers, opts)

done = ->
  workers\addjob (-> dataLoader\terminate!), ->
  workers\terminate!
  os.exit!

-- set up callbacks
cbq = with CallbackQueue(opts.startiter)
  \add cb: done, iter: opts.niters > 0 and opts.niters or math.huge, priority: -math.huge
  \add cb: val, interval: opts.valfreq, iter: opts.valfreq, priority: math.huge
  \add cb: snap, interval: opts.savefreq, iter: opts.savefreq if opts.savefreq > 0

collectgarbage!

-- val!
while #cbq > 0
  train! for t=1,cbq\waitTime!
  workers\synchronize!
  cbq\advance!
  cb! for cb in cbq\pull!
