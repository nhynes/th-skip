require 'paths'
_ = require 'moses'

DataLoader = torch.class('DataLoader')

UNK, EOR, EOS = 1, 2, 3

groupByLen = (data) ->
  slens = data.slens

  indsByLen = {}
  for i=1,slens\size(1)
    slen = slens[i]

    ibrl = indsByLen[slen] or {}
    ibrl[#ibrl+1] = i
    indsByLen[slen] = ibrl

  lengths = _.keys indsByLen
  for len, inds in pairs indsByLen
    lengths[#lengths+1] = len
    indsByLen[len] = torch.LongTensor(inds)
  table.sort lengths

  lenFreqs = torch.zeros(#lengths)
  lenFreqs[i] = indsByLen[lengths[i]]\size(1) for i=1,#lengths

  data.lengths = lengths
  data.indsByLen = indsByLen
  data.lenFreqs = lenFreqs
  data.isBoundary = {data.rbps[i], true for i=1,data.rbps\size(1)}

DataLoader.__init = (dataTrain, dataVal, opts) =>
  @batchSize = opts.batchSize
  @dataTrain = dataTrain
  @dataVal = dataVal

  @vocabSize = opts.vocabSize

  groupByLen(dataTrain)
  groupByLen(dataVal)

DataLoader.makebatch = (partition='train', seed) =>
  data = partition == 'val' and @dataVal or @dataTrain

  [==[
    1. pick a sentence length
    <until full minibatch>
    2. grab a sentence, add </s> and </s>
    3. grab prev and next sentences
      if sentence is on a recipe boundary (check rbp), prev sentence is </r>
      if next sentence is on a boundary, next sentence is </r>
    4. add leading and trailing </s>
  ]==]

  randState = nil
  if seed ~= nil
    randState = torch.getRNGState!
    torch.manualSeed(seed)

  sentlen = data.lengths[torch.multinomial(data.lenFreqs, 1)[1]]
  sentlenInds = data.indsByLen[sentlen]
  batchSize = math.min(sentlenInds\size(1), @batchSize)

  selIndInds = torch.randperm(sentlenInds\size(1))\sub(1, batchSize)\long!

  maxSentLen = data.lengths[#data.lengths] + 2 -- +2 for </s> ... </s>

  -- batchIds = torch.CharTensor(batchSize, 11) -- bin search rbps for selInd
  batchSents = torch.LongTensor(batchSize, sentlen+2)\zero! -- +1 for </s> ... </s>
  batchPrevSents = torch.LongTensor(batchSize, maxSentLen)\zero!
  batchNextSents = torch.LongTensor(batchSize, maxSentLen)\zero!

  toks = data.toks
  nsents = toks\size(1)
  isBoundary = data.isBoundary
  slens = data.slens

  -- length of sentences including trailing </s> but not leading </s>
  maxPrev = 0
  maxNext = 0

  selInds = sentlenInds\index(1, selIndInds)

  for i=1,batchSize
    selInd = selInds[i]

    batchSents[i]\sub(2, -2)\copy(toks[selInd]\sub(1, sentlen))

    prevSent = batchPrevSents[i]\sub(2, -1)
    nextSent = batchNextSents[i]\sub(2, -1)

    if isBoundary[selInd] or selInd == 1
      prevSent[1] = EOR
      prevSent[2] = EOS
      maxPrev = math.max(maxPrev, 1)
    else
      slen = slens[selInd-1]
      prevSent\sub(1, -2)\copy(toks[selInd-1])
      prevSent[slen+1] = EOS
      maxPrev = math.max(maxPrev, slen)

    if isBoundary[selInd+1] or selInd == nsents
      nextSent[1] = EOR
      nextSent[2] = EOS
      maxNext = math.max(maxNext, 1)
    else
      slen = slens[selInd+1]
      nextSent\sub(1, -2)\copy(toks[selInd+1])
      nextSent[slen+1] = EOS
      maxNext = math.max(maxNext, slen)

  batchSents\select(2, 1)\fill(EOS)
  batchSents\select(2, sentlen+2)\fill(EOS)

  batchPrevSents = batchPrevSents\narrow(2, 1, maxPrev+2)
  batchPrevSents\select(2, 1)\fill(EOS)
  batchNextSents = batchNextSents\narrow(2, 1, maxNext+2)
  batchNextSents\select(2, 1)\fill(EOS)

  collectgarbage!

  torch.setRNGState(randState) if randState ~= nil

  batchSents, batchPrevSents, batchNextSents

DataLoader.partitionSize = (partition='train') =>
  ds = partition == 'train' and @dataTrain or @dataVal
  ds.toks\size(1)

DataLoader.terminate = =>
