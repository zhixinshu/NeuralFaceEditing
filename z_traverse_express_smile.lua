------------------------------------------------------------
--- Code for 
-- Shu et al., Neural Face Editing with Intrinsic Image Disentangling, CVPR 2017.
------------------------------------------------------------


require 'hdf5'
require 'nngraph'
require 'cudnn'
require 'torch'
require 'nn'
require 'cunn'
require 'optim'
require 'image'
require 'pl'
require 'paths'

require 'modules/SHShading'
require 'modules/SACompose'
require 'modules/SHPartialShading'
require 'modules/SHPartialShadingRGB'
require 'modules/SHPartialShadingRGB_bw'
require 'modules/TVLoss'
require 'modules/TVCriterion'
require 'modules/TVSelfCriterion'
require 'modules/TVSelfPartialCriterion'
require 'modules/RangeSelfCriterion'
require 'modules/SmoothSelfCriterion'
require 'modules/SmoothSelfPartialCriterion'
require 'modules/UniLengthCriterion'
require 'modules/PerElementNorm'
require 'modules/BatchWhiteShadingCriterion'
require 'modules/MarginNegMSECriterion'
require 'modules/BatchWhiteShadingCriterion2'
require 'modules/LightCoeffCriterion'
require 'modules/PartialSimNCriterion'
require 'modules/FBMCompose'
require 'modules/MaskedReconCriterion'


ok, disp = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end
formation = require 'dipface_express'

----------------------------------------------------------------------
-- parse command-line optionsls
opt = lapp[[
  -s,--save          (default 'z_traverse_express_5_smile')      subdirectory to save logs
  --editsave         (default '')      subdirectory to save editing
  --saveFreq         (default 1)          save every saveFreq epochs
  -n,--network       (default 'models/dipface_express_pretrain_CelebA.t7')          reload pretrained network
  -p,--plot                                 plot while training
  -r,--learningRate  (default 0.001)        learning rate
  -b,--batchSize     (default 100)          batch size
  -m,--momentum      (default 0)            momentum, for SGD only
  --coefL1           (default 0)            L1 penalty on the weights
  --coefL2           (default 0)            L2 penalty on the weights
  -t,--threads       (default 4)            number of threads
  -g,--gpu           (default 2)            gpu to run on (default cpu)
  -d,--noiseDim      (default 512)          dimensionality of noise vector
  --K                (default 1)            number of iterations to optimize D for
  -w, --window       (default 3)            windsow id of sample image
  --scale            (default 64)           scale of images to train on
  --weight_1         (default 100)
  --weight_2		 (default 1)
  --z_dim            (default 128)          dimensionality of z code
  --zA_dim           (default 128)         dimensionality of A code
  --zN_dim           (default 128)         dimensionality of N code
  --zL_dim           (default 10)         dimensionality of L code
  --zB_dim           (default 128)         dimensionality of B code
  --zM_dim           (default 32)         dimensionality of M code
  --dz_dim           (default 128)          dimensionality of dz code
  --margin           (default 20)           value of margin
]]

paths.mkdir(opt.save)

nw1 = opt.weight_1/(opt.weight_1 + opt.weight_2)
nw2 = opt.weight_2/(opt.weight_1 + opt.weight_2)


if opt.gpu < 0 or opt.gpu > 3 then opt.gpu = false end

print(opt)

-- fix seed
torch.manualSeed(1)

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

local function sleep(n)
  os.execute("sleep " .. tonumber(n))
end

if opt.gpu then
  cutorch.setDevice(opt.gpu + 1)
  print('<gpu> using device ' .. opt.gpu)
  torch.setdefaulttensortype('torch.CudaTensor')
else
  torch.setdefaulttensortype('torch.FloatTensor')
end

opt.geometry = {3, opt.scale, opt.scale}
--opt.geometry = {6, opt.scale, opt.scale}

local input_sz = opt.geometry[1] * opt.geometry[2] * opt.geometry[3]

------------------------------------------
-- load pre trained network
------------------------------------------  
print('<trainer> reloading previously trained network: ' .. opt.network)
tmp = torch.load(opt.network)
model_D = tmp.D
model_Enc = tmp.Enc
model_Dec = tmp.Dec

if opt.gpu then
  print('Copy model to gpu')
  model_D:cuda()
  model_Enc:cuda()
  model_Dec:cuda()
end

----------------------------------------------------------------------
-- Get examples to plot
function getSamples(dataset, N, fixid)
  local numperclass = numperclass or 10
  local N = N or 8
  local fixid = fixid or false
  --local noise_inputs = torch.Tensor(N, opt.noiseDim)
  -- Generate samples
  local inputs_samples_img = torch.Tensor(N, opt.geometry[1], opt.geometry[2], opt.geometry[3])
  local inputs_samples_mask = torch.Tensor(N, opt.geometry[1], opt.geometry[2], opt.geometry[3])
  --local inputs_samples_bias = torch.Tensor(N, opt.geometry[1], opt.geometry[2], opt.geometry[3])
  math.randomseed( os.time() )
  for i = 1,N do
    local idx = math.random(dataset:size()[1])
    if fixid then
        idx = i
    end
    local sample = dataset[idx]
    local sample_img = sample:narrow(1,1,3)
    local sample_mask = sample:narrow(1,7,1)
    --local sample_bias = sample_img:clone()
    sample_mask = torch.repeatTensor(sample_mask,3,1,1)
    --sample_bias = sample_bias:fill(1)
    inputs_samples_img[i] = sample_img:clone()
    inputs_samples_mask[i] = sample_mask:clone()
    --inputs_samples_bias[i] = sample_bias:clone()
  end 
  local zA_sample, zN_sample, zL_sample = unpack(model_Enc:forward(inputs_samples_img))
  local synth_samples,A_samples,N_samples,Lr_samples,Lg_samples,Lb_samples,S_samples = unpack(model_Dec:forward({zA_sample, zN_sample, zL_sample,inputs_samples_mask}))
  synth_samples = nn.HardTanh():forward(synth_samples)
  A_samples = nn.HardTanh():forward(A_samples)
  N_samples = nn.HardTanh():forward(N_samples)
  S_samples = nn.Tanh():forward(S_samples)
  N_samples = (N_samples+1)*0.5

  local to_plot_0 = {}
  local to_plot_1 = {}
  local to_plot_2 = {}
  local to_plot_3 = {}
  local to_plot_4 = {}
  for i=1,N do
    to_plot_0[#to_plot_0 + 1] = inputs_samples_img[i]:float()
    to_plot_1[#to_plot_1 + 1] = synth_samples[i]:float()
    to_plot_2[#to_plot_2 + 1] = A_samples[i]:float()
    to_plot_3[#to_plot_3 + 1] = N_samples[i]:float()
    to_plot_4[#to_plot_4 + 1] = S_samples[i]:float()
  end

  return to_plot_0, to_plot_1, to_plot_2, to_plot_3, to_plot_4
end



--%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function zBarycenterTraverse(input, srcData, tgtData)
    local srcBarycenter = torch.mean(srcData, 1)
    local tgtBarycenter = torch.mean(tgtData, 1)
    local arrow = tgtBarycenter - srcBarycenter
    local arrowArray = torch.repeatTensor(arrow,input:size()[1],1)
    local output  = input + arrowArray
    return output
end

--%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plotImageMatrix(imagedata, folder, name)
    torch.setdefaulttensortype('torch.CudaTensor')
    local plot_data = {}
    print(imagedata:size())
    imagedata = nn.HardTanh():forward(imagedata)
    torch.setdefaulttensortype('torch.FloatTensor')
    for i =1, imagedata:size()[1] do
        plot_data[#plot_data + 1] = imagedata[i]:float()
    end
    local plot_formatted = image.toDisplayTensor({input = plot_data, nrow=10})
    plot_formatted:float()
    image.save(folder .. name ..'.png', plot_formatted)
    torch.setdefaulttensortype('torch.CudaTensor')
end
--%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("Reading positive training data .. ")
local phdf5 = hdf5.open('data/zx_7_d10_inmc_train_smileP.hdf5', 'r')
posData = phdf5:read('zx_7'):all()
phdf5:close()

print("Reading negative training data .. ")
local nhdf5 = hdf5.open('data/zx_7_d10_inmc_train_smileN.hdf5', 'r')
negData = nhdf5:read('zx_7'):all()
nhdf5:close()

print("Reading positive testing data .. ")
local phdf5 = hdf5.open('data/zx_7_d10_inmc_test_smileP.hdf5', 'r')
test_posData = phdf5:read('zx_7'):all()
phdf5:close()

print("Reading negative testing data .. ")
local nhdf5 = hdf5.open('data/zx_7_d10_inmc_test_smileN.hdf5', 'r')
test_negData = nhdf5:read('zx_7'):all()
nhdf5:close()
-------------------------------------------------------------------------------------------
-- compute the Zs of positive data and negative data
local posClusterSize = 2000
local posBatchSize = 100
local negClusterSize = 2000
local negBatchSize = 100
local posInputBatch = torch.Tensor(posBatchSize, opt.geometry[1], opt.geometry[2], opt.geometry[3])
local negInputBatch = torch.Tensor(negBatchSize, opt.geometry[1], opt.geometry[2], opt.geometry[3])

--------------------------------------------------------------------------------------------
--sample positive data zs
local pos_zA = torch.Tensor( posClusterSize, opt.zA_dim)
local pos_zN = torch.Tensor( posClusterSize, opt.zN_dim)
local pos_zL = torch.Tensor( posClusterSize, opt.zL_dim)
local pos_zB = torch.Tensor( posClusterSize, opt.zB_dim)
local pos_zM = torch.Tensor( posClusterSize, opt.zM_dim)
for i = 1, posClusterSize, posBatchSize do
    for j = 1,posBatchSize do
        local idx = math.random(posData:size()[1])
        local sample = posData[idx] -- sample data point
        local sample_img = sample:narrow(1,1,3) -- the img in the data point, firt 3 dimenions
        posInputBatch[j] = sample_img:clone()
    end
    zA, zN, zL, zB, zM = unpack(model_Enc:forward(posInputBatch))
    pos_zA[{{i, i+posBatchSize-1},{}}] = zA:clone()
    pos_zN[{{i, i+posBatchSize-1},{}}] = zN:clone()
    pos_zL[{{i, i+posBatchSize-1},{}}] = zL:clone()
    pos_zB[{{i, i+posBatchSize-1},{}}] = zB:clone()
    pos_zM[{{i, i+posBatchSize-1},{}}] = zM:clone()
end
--sample negative data zs
local neg_zA = torch.Tensor( negClusterSize, opt.zA_dim)
local neg_zN = torch.Tensor( negClusterSize, opt.zN_dim)
local neg_zL = torch.Tensor( negClusterSize, opt.zL_dim)
local neg_zB = torch.Tensor( negClusterSize, opt.zB_dim)
local neg_zM = torch.Tensor( negClusterSize, opt.zM_dim)
for i = 1, negClusterSize, negBatchSize do
    for j = 1,negBatchSize do
        local idx = math.random(negData:size()[1])
        local sample = negData[idx] -- sample data point
        local sample_img = sample:narrow(1,1,3) -- the img in the data point, firt 3 dimenions
        negInputBatch[j] = sample_img:clone()
    end
    zA, zN, zL, zB, zM = unpack(model_Enc:forward(negInputBatch))
    neg_zA[{{i, i+negBatchSize-1},{}}] = zA:clone()
    neg_zN[{{i, i+negBatchSize-1},{}}] = zN:clone()
    neg_zL[{{i, i+negBatchSize-1},{}}] = zL:clone()
    neg_zB[{{i, i+negBatchSize-1},{}}] = zB:clone()
    neg_zM[{{i, i+negBatchSize-1},{}}] = zM:clone()
end

--------------------------------------------------------------------------------------------
-- pret (pre-traversal), the input of the traversal, on the z manifold
local n_pret = 100
local pret_img = torch.Tensor(n_pret, opt.geometry[1], opt.geometry[2], opt.geometry[3])
local pret_mask = torch.Tensor(n_pret, opt.geometry[1], opt.geometry[2], opt.geometry[3])
-- sample images from negative data
for i = 1, n_pret do
    --local idx = math.random(test_negData:size()[1])
    local idx = i
    local sample = test_negData[idx]
    local sample_img = sample:narrow(1,1,3)
    local sample_mask = sample:narrow(1,7,1)
    sample_mask = torch.repeatTensor(sample_mask,3,1,1)
    pret_img[i] = sample_img:clone()
    pret_mask[i] = sample_mask:clone()
end
pret_zA, pret_zN, pret_zL, pret_zB, pret_zM = unpack(model_Enc:forward(pret_img))
local pret_synth, pret_A, pret_N, pret_Lr, pret_Lg, pret_Lb, pret_S, pret_Nnm, pret_B, pret_M, pret_final = unpack(model_Dec:forward({pret_zA, pret_zN, pret_zL, pret_zB, pret_zM, pret_mask}))
--plot  pres
plotImageMatrix(pret_img, opt.save, '/pret_image')
plotImageMatrix(pret_synth, opt.save , '/pret_synth')
plotImageMatrix(pret_A, opt.save , '/pret_A')
plotImageMatrix(pret_N, opt.save , '/pret_N')
plotImageMatrix(pret_S, opt.save , '/pret_S')
plotImageMatrix(pret_B, opt.save , '/pret_B')
plotImageMatrix(pret_M, opt.save , '/pret_M')
plotImageMatrix(pret_final, opt.save , '/pret_final')
--------------------------------------------------------------------------------------------
-- obtain pt (post-traversal), the output of the traversal, on the z manifold
torch.setdefaulttensortype('torch.CudaTensor')
pt_zA = zBarycenterTraverse(pret_zA, neg_zA, pos_zA) 
pt_zN = zBarycenterTraverse(pret_zN, neg_zN, pos_zN) 
pt_zL = zBarycenterTraverse(pret_zL, neg_zL, pos_zL) 
pt_zB = zBarycenterTraverse(pret_zB, neg_zB, pos_zB) 
pt_zM = zBarycenterTraverse(pret_zM, neg_zM, pos_zM) 
--------------------------------------------------------------------------------------------
-- reconstruct image of postt
-- synthetic image(syn), albedo(A), normal(N), light(Lr, Lg, Lb), shading(S)
local pt_mask = pret_mask:clone()
local pt_synth, pt_A, pt_N, pt_Lr, pt_Lg, pt_Lb, pt_S, pt_Nnm, pt_B, pt_M, pt_final = unpack(model_Dec:forward({pt_zA, pt_zN, pret_zL, pret_zB, pret_zM, pt_mask}))
--plot  posts
plotImageMatrix(pt_synth, opt.save , '/pt_synth')
plotImageMatrix(pt_A, opt.save , '/pt_A')
plotImageMatrix(pt_N, opt.save , '/pt_N')
plotImageMatrix(pt_S, opt.save , '/pt_S')
plotImageMatrix(pt_B, opt.save , '/pt_B')
plotImageMatrix(pt_M, opt.save , '/pt_M')
plotImageMatrix(pt_final, opt.save , '/pt_final')

