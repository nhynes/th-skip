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
UNK, EOR, EOS = 1, 2, 3

torch.setdefaulttensortype 'torch.FloatTensor'

---------------------------------------------------------------------------------------

cmd = with torch.CmdLine!
  \option '-encs', '', 'path to encoded sent vecs'
  \option '-model', '', 'path to model'
  \option '-vocab', PROJ_ROOT..'/data/instructions_w2v_vocab.txt', 'path to model'
  \option '-batchSize', 1024, 'max number of sentences to encode at once'
  \option '-out', PROJ_ROOT..'/data/decoded_sents', 'prefix to save decoded sentences'
  \option '-samples', 1, 'number of decodings for each sentence'
  \option '-greedy', 0, 'greedy or stochastic (greedy -> samples=1)'
opts = cmd\parse arg

---------------------------------------------------------------------------------------
-- Load decoder
---------------------------------------------------------------------------------------

dofile(PROJ_ROOT..'/model/init.moon').init{decoding: opts.model}
{:model, opts: modelOpts} = torch.load(opts.model)

maxSentlen = modelOpts.sentlen

decoder = with model\get(1)
  with .model
    \add nn.Select(1, 1)
    \add nn.SoftMax!
  \cuda!
  \evaluate!

---------------------------------------------------------------------------------------
-- Load Data
---------------------------------------------------------------------------------------

-- load vocab
fVocab = io.open(opts.vocab)
vocab = fVocab\read '*all'
fVocab\close!
i2w = {'UNK', '</r>'}
i2w[#i2w+1] = word for word in string.gmatch(vocab, '[^\t\n]+')

-- load encoded sentences
encs = torch.load(opts.encs).encs

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
opts.samples = opts.greedy == 1 and 1 or opts.samples

bbs = batchSize * opts.samples
stDim = encs\size(2)
gpuEncs = torch.CudaTensor(bbs, stDim)
nextToks = torch.CudaTensor(bbs, 1)
gpuSents = torch.CudaTensor(bbs, maxSentlen)
sents = torch.LongTensor(bbs, maxSentlen)

decRNN = decoder.rnn
decRNN.hiddenInput = torch.CudaTensor(1, bbs, decRNN.hiddenSize)
decRNN.cellInput = torch.CudaTensor(1, bbs, decRNN.hiddenSize)

N = encs\size(1)
for i=1,encs\size(1),batchSize
  decRNN.hiddenInput\zero!
  decRNN.cellInput\zero!

  bs = math.min(batchSize, N-i+1)
  with gpuEncs\resize(bs * opts.samples, stDim)
    \copy(encs\narrow(1, i, bs)\view(bs, 1, stDim)\expand(bs, opts.samples, stDim))

  nextToks\fill(EOS)
  for t=1,maxSentlen
    preds = decoder\forward({nextToks, gpuEncs})
    if opts.greedy == 1
      p, mltoks = torch.sort(preds, 2, true)
      nextToks\copy(mltoks\select(2, 1))
    else
      torch.multinomial(nextToks, preds, 1)
    gpuSents\select(2, t)\copy(nextToks)

    decRNN.hiddenInput\copy(decRNN.hiddenOutput)
    decRNN.cellInput\copy(decRNN.cellOutput)

  sents\copy(gpuSents)
  for n=1,sents\size(1)
    if n % opts.samples == 1
      print('\n'..(i+math.floor(n/opts.samples))..':')
    print convertI2W(sents[n])
