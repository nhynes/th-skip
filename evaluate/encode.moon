require 'torch'
require 'cudnn'
require 'hdf5'
require 'dpnn'
require 'xlua'
require 'sys'
_ = require 'moses'
import thisfile from require 'paths'
import dofile from require 'moonscript'

PROJ_ROOT = thisfile '../'
UNK, SOR, EOR, EOS = 1, 2, 3, 4
VERIFY_IND = 1024

torch.setdefaulttensortype 'torch.FloatTensor'

---------------------------------------------------------------------------------------

cmd = with torch.CmdLine!
  \option '-data', PROJ_ROOT..'/data/dataset.h5', 'path to dataset'
  \option '-partition', 'test', '(train|val|test)'
  \option '-model', PROJ_ROOT..'/snaps/model.t7', 'path to model'
  \option '-batchSize', 4096, 'max number of sentences to encode at once'
  \option '-out', 'encoded_sents', 'prefix to save encoded sentences'
opts = cmd\parse arg

---------------------------------------------------------------------------------------
-- Load encoder
---------------------------------------------------------------------------------------

dofile PROJ_ROOT..'/model/init.moon'
snap = torch.load(opts.model)
{:model, opts: modelOpts} = snap
encoder = model\get(1)\get(2)\cuda!
encoder\evaluate!

vocabSize = modelOpts.vocabSize + 1
assert vocabSize == encoder\get(1)\get(1).weight\size(1)
encDim = modelOpts.dim*2 -- bidirectional

model = nil
collectgarbage!

---------------------------------------------------------------------------------------
-- Load Data
---------------------------------------------------------------------------------------

groupByLen = (data) ->
  slens = data.slens

  indsByLen = {}
  for i=1,slens\size(1)
    slen = slens[i]

    ibrl = indsByLen[slen] or {}
    ibrl[#ibrl+1] = i
    indsByLen[slen] = ibrl

  lengths = _.keys indsByLen
  table.sort lengths

  lenFreqs = torch.zeros(#lengths)
  lenFreqs[i] = #indsByLen[lengths[i]] for i=1,#lengths -- freq of each index -> len

  data.lengths = lengths
  data.indsByLen = indsByLen

dsH5 = hdf5.open(opts.data)
toks = dsH5\read('/toks_'..opts.partition)\all!
toks\maskedFill(toks\gt(vocabSize), UNK)
data =
  toks: toks
  slens: dsH5\read('/slens_'..opts.partition)\all!
dsH5\close!
groupByLen(data)

---------------------------------------------------------------------------------------
-- Encode
---------------------------------------------------------------------------------------

toks = data.toks
N = toks\size(1)

encs = torch.Tensor(N, encDim)
inds = torch.LongTensor(N)
batchSents = torch.ShortTensor!
batchSentsIdx = torch.ShortTensor!
gpuSents = torch.CudaTensor!

n = 0
for sentlen in *_.reverse(data.lengths)
  sentlenInds = data.indsByLen[sentlen]
  strides = torch.LongStorage{sentlen+1, 1}
  for batchInds in _.partition(sentlenInds, opts.batchSize)
    batchSize = #batchInds

    selInds = torch.LongTensor(batchInds)

    batchSents\resize(batchSize, sentlen+1)
    -- avoid double-allocation when appending EOS
    batchSentsIdx\set batchSents\storage!, 1,
      torch.LongStorage{batchSize, toks\size(2)},
      strides
    batchSentsIdx\index(toks, 1, selInds) -- copy
    batchSents\select(2, sentlen+1)\fill(EOS)
    -- batchSents[{i, {1, sentlen}}] = toks[{batchInds[i], {1, sentlen}}] for i=1,batchSize

    gpuSents\resize(batchSents\size!)\copy(batchSents)
    encoder\forward(gpuSents)

    encs\narrow(1, n+1, batchSize)\copy(encoder.output)
    inds\narrow(1, n+1, batchSize)\copy(selInds)

    n += batchSize
    xlua.progress(n, N)

_, sortInds = torch.sort(inds)
encs = encs\index(1, sortInds)
outfile = opts.out..'_'..opts.partition..'.t7'
torch.save(outfile, {encs: encs, opts: opts})
