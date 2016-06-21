import dofile from require 'moonscript'

require 'torch'
require 'cutorch'
require 'drivers.CallbackQueue'

require 'model.init'

args = dofile 'args.moon'
loader = dofile 'loader/init.moon'
drivers = dofile 'drivers/init.moon'

opts = args.parse arg

torch.setdefaulttensortype 'torch.FloatTensor'
torch.manualSeed opts.seed
math.randomseed opts.seed

cutorch.setDevice opts.gpu+1

workers = loader.init opts

model = nil
if paths.filep opts.loadsnap
  print 'Loading model from '..opts.loadsnap
  model = torch.load opts.loadsnap
else
  model = Model(opts)
model\cuda!

train, val, snap = drivers.init model, workers, opts

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
