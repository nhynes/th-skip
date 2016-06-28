require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'dpnn'
require 'hdf5'
require 'xlua'
_ = require 'moses'
import thisfile from require 'paths'
import dofile from require 'moonscript'

PROJ_ROOT = thisfile '../'
UNK, SOR, EOR, SOS, EOS = 1, 2, 3, 4, 5

torch.setdefaulttensortype 'torch.FloatTensor'

---------------------------------------------------------------------------------------

cmd = with torch.CmdLine!
  \option '-encs', PROJ_ROOT..'/data/encoded_sents.t7', 'path to encoded sent vecs'
  \option '-model', PROJ_ROOT..'/snaps/decoder.t7', 'path to model'
  \option '-vocab', PROJ_ROOT..'/data/instructions_w2v_vocab.txt', 'path to model'
  \option '-batchSize', 3, 'max number of sentences to encode at once'
  \option '-out', PROJ_ROOT..'/data/decoded_sents', 'prefix to save decoded sentences'
  \option '-beam', 1, 'beam search width'
opts = cmd\parse arg

---------------------------------------------------------------------------------------
-- Load decoder
---------------------------------------------------------------------------------------

print 'loading model'
model = dofile PROJ_ROOT..'/model/init.moon'
model.init{decoding: opts.model}
{:model, opts: modelOpts} = torch.load(opts.model)

maxSentlen = modelOpts.sentlen

{decoder: dec, :lut} = model\get(1)
with dec
  \remove!
  \add cudnn.SpatialSoftMax!

decoder = with nn.Sequential!
  \add with nn.ParallelTable!
    \add lut
    \add nn.Identity!
  \add dec
  \cuda!
  \evaluate!

decRNN = dec\get(2)
assert torch.isTypeOf(decRNN, cudnn.RNN)
-- decRNN.hiddenInput = decRNN.hiddenOutput
-- decRNN.cellInput = decRNN.cellOutput

model = nil

---------------------------------------------------------------------------------------
-- Load Data
---------------------------------------------------------------------------------------

-- load vocab
fVocab = io.open(opts.vocab)
vocab = fVocab\read '*all'
fVocab\close!
i2w = {'UNK', '<r>', '</r>', '<s>'}
i2w[#i2w+1] = word for word in string.gmatch(vocab, '[^\t\n]+')

-- load encoded sentences
print 'loading sentences'
encs = torch.load(opts.encs)

convertI2W = (v) ->
  toks = {}
  for i=1,v\size(1)
    tok = v[i]
    ctok = i2w[tok]
    if tok == EOS
      break
    toks[#toks+1] = ctok
    if ctok == '.'
      break
  _.concat(toks, ' ')

---------------------------------------------------------------------------------------
-- Decode
---------------------------------------------------------------------------------------

collectgarbage!

batchSize = opts.batchSize

bbs = batchSize * opts.beam
stDim = encs\size(2)
gpuEncs = torch.CudaTensor(bbs, stDim)
nextToks = torch.CudaTensor(bbs, 1)
gpuSents = torch.CudaTensor(bbs, maxSentlen)
sents = torch.LongTensor(bbs, maxSentlen)

decRNN.hiddenInput = torch.CudaTensor(1, bbs, decRNN.hiddenSize)
decRNN.cellInput = torch.CudaTensor(1, bbs, decRNN.hiddenSize)

-- stochastic impl
N = encs\size(1)
for i=1,encs\size(1),batchSize
  if i > 10
    break

  decRNN.hiddenInput\zero!
  decRNN.cellInput\zero!

  batchEncs = encs\narrow(1, i, math.min(batchSize, N-i+1))
  batchEncs = batchEncs\view(batchSize, 1, stDim)\expand(batchSize, opts.beam, stDim)
  gpuEncs\copy(batchEncs)

  nextToks\fill(SOS)
  for t=1,maxSentlen
    preds = decoder\forward({nextToks, gpuEncs})\select(2, 1)\t!
    -- torch.multinomial(nextToks, preds, 1)
    s, amax = torch.sort(preds, 2, true)
    nextToks\copy(amax\select(2, 1))
    gpuSents\select(2, t)\copy(nextToks)

    decRNN.hiddenInput\copy(decRNN.hiddenOutput)
    decRNN.cellInput\copy(decRNN.cellOutput)

  sents\copy(gpuSents)
  for n=1,sents\size(1)
    print(convertI2W(sents\select(1, n)))
    if n % opts.beam == 0
      print('')
