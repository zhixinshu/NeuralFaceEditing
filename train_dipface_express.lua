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
require 'modules/BatchWhiteShadingCriterion2'
require 'modules/MarginNegMSECriterion'
require 'modules/LightCoeffCriterion'
require 'modules/PartialSimNCriterion'
require 'modules/FBMCompose'
require 'modules/MaskedReconCriterion'

ok, disp = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end
formation = require 'dipface_express'

--models/dipface_hardnn_nounpool_batchwhite_ebadv_ct4_e217.net
----------------------------------------------------------------------
-- parse command-line options
opt = lapp[[
  -s,--save          (default "logs_dipface_express")      subdirectory to save logs
  --saveFreq         (default 1)          save every saveFreq epochs
  -n,--network       (default './models/dipface_express_pretrain.t7')          reload pretrained network
  -p,--plot                                plot while training
  -r,--learningRate  (default 0.001)        learning rate
  -b,--batchSize     (default 100)         batch size
  -m,--momentum      (default 0)           momentum, for SGD only
  --coefL1           (default 0)           L1 penalty on the weights
  --coefL2           (default 0)           L2 penalty on the weights
  -t,--threads       (default 4)           number of threads
  -g,--gpu           (default 0)           gpu to run on (default cpu)
  -d,--noiseDim      (default 512)         dimensionality of noise vector
  --K                (default 1)           number of iterations to optimize D for
  -w, --window       (default 3)           windsow id of sample image
  --scale            (default 64)          scale of images to train on
  --weight_1         (default 100)
  --weight_2		     (default 1)
  --z_dim            (default 128)         dimensionality of z code
  --zA_dim           (default 128)         dimensionality of A code
  --zN_dim           (default 128)         dimensionality of N code
  --zL_dim           (default 10)         dimensionality of L code
  --zB_dim           (default 128)         dimensionality of B code
  --zM_dim           (default 32)         dimensionality of M code
  --dz_dim           (default 128)       dimensionality of dz code
  --margin           (default 20)           value of margin
]]

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

if opt.network == '' then
----------------------------------------------------------------------
  -- define D network (normal GAN)
  ----------------------------------------------------------------------
  --d_input = nn.Identity()()
  --ld0 = cudnn.SpatialConvolution(3, 32, 5, 5, 1, 1, 2, 2)(d_input)
  --ld0 = cudnn.SpatialMaxPooling(2,2)(ld0)
  --ld0 = cudnn.ReLU(true)(ld0)
  --ld0 = nn.SpatialDropout(0.2)(ld0)
  --ld0 = cudnn.SpatialConvolution(32, 64, 5, 5, 1, 1, 2, 2)(ld0)
  --ld0 = cudnn.SpatialMaxPooling(2,2)(ld0)
  --ld0 = cudnn.ReLU(true)(ld0)
  --ld0 = nn.SpatialDropout(0.2)(ld0)
  --ld0 = cudnn.SpatialConvolution(64, 96, 5, 5, 1, 1, 2, 2)(ld0)
  --ld0 = cudnn.ReLU(true)(ld0)
  --ld0 = cudnn.SpatialMaxPooling(2,2)(ld0)
  --ld0 = nn.SpatialDropout(0.2)(ld0)
  --ld0 = nn.Reshape(8*8*96)(ld0)
  --ld0 = nn.Linear(8*8*96,1024)(ld0)
  --ld0 = cudnn.ReLU(true)(ld0)
  --ld0 = nn.Dropout()(ld0)
  --ld0 = nn.Linear(1024,1)(ld0)
  --d_output = nn.Sigmoid()(ld0)

  --model_D = nn.gModule({d_input},{d_output})

  ----------------------------------------------------------------------
  -- define D network, a Enc-Dec network (EBGAN)
  ----------------------------------------------------------------------

  d_input = nn.Identity()()

  -- D-encoder
  d_enc = cudnn.SpatialConvolution(3, 96, 5, 5, 1, 1, 2, 2)(d_input)
  d_enc = cudnn.SpatialMaxPooling(2,2)(d_enc)
  d_enc = nn.Threshold(0, 1e-6)(d_enc)
  d_enc = cudnn.SpatialConvolution(96, 48, 5, 5, 1, 1, 2, 2)(d_enc)
  d_enc = cudnn.SpatialMaxPooling(2,2)(d_enc)
  d_enc = nn.Threshold(0, 1e-6)(d_enc)
  d_enc = cudnn.SpatialConvolution(48, 24, 5, 5, 1, 1, 2, 2)(d_enc)
  d_enc = cudnn.SpatialMaxPooling(2,2)(d_enc)
  d_enc = nn.Threshold(0, 1e-6)(d_enc)
  d_enc = nn.Reshape(24*8*8)(d_enc)
  d_enc = nn.Linear(24*8*8,opt.dz_dim)(d_enc)
  -- D code
  local d_z = nn.Sigmoid()(d_enc)
  -- D-decoder
  d_dec = nn.Identity()(d_z)
  d_dec = nn.Linear(opt.dz_dim, 24*8*8)(d_dec)
  d_dec = nn.Threshold(0, 1e-6)(d_dec)
  d_dec = nn.Reshape(24, 8, 8)(d_dec)
  d_dec = nn.SpatialUpSamplingNearest(2)(d_dec)
  d_dec = cudnn.SpatialConvolution(24, 48, 5, 5, 1, 1, 2, 2)(d_dec)
  d_dec = nn.Threshold(0, 1e-6)(d_dec)
  d_dec = nn.SpatialUpSamplingNearest(2)(d_dec)
  d_dec = cudnn.SpatialConvolution(48, 96, 5, 5, 1, 1, 2, 2)(d_dec)
  d_dec = nn.Threshold(0, 1e-6)(d_dec)
  d_dec = nn.SpatialUpSamplingNearest(2)(d_dec)
  d_dec = cudnn.SpatialConvolution(96, 96, 5, 5, 1, 1, 2, 2)(d_dec)
  d_dec = nn.Threshold(0, 1e-6)(d_dec)
  d_dec = cudnn.SpatialConvolution(96, 3, 3, 3, 1, 1, 1, 1)(d_dec)
  d_output =  nn.Threshold(0, 1e-6)(d_dec)
  model_D = nn.gModule({d_input},{d_output})


  ----------------------------------------------------------------------
  -- define G network
  -- define the encoder-decoder that generates output for synthesis layer
  -- input: I, a 3xWxH image tensor
  -- output: 
  -- output[1] : A, a 3xWxH tensor approximates albedo
  -- output[2] : N, a 1xWxH tensor approximates the normal
  -- output[3] : L, a 9x1 vector approximates the spherical harmonics parameters  
      
  ----------------------------------------------------------------
      -- the encoder(s)
  -------------------------------------------------------
  e_input = nn.Identity()()

  encoder = cudnn.SpatialConvolution(3, 96, 5, 5, 1, 1, 2, 2)(e_input)
  local mp1 = nn.SpatialMaxPooling(2, 2)
  mp1_shell = nn.Sequential()
  mp1_shell:add(mp1)
  encoder_mp1 = mp1_shell(encoder)
  encoder_mp1 = nn.Threshold(0, 1e-6)(encoder_mp1)
  encoder_mp1 = cudnn.SpatialConvolution(96, 48, 5, 5, 1, 1, 2, 2)(encoder_mp1)
  local mp2 = nn.SpatialMaxPooling(2, 2)
  mp2_shell = nn.Sequential()
  mp2_shell:add(mp2)
  encoder_mp2 = mp2_shell(encoder_mp1)
  encoder_mp2 = nn.Threshold(0, 1e-6)(encoder_mp2)
  encoder_mp2 = cudnn.SpatialConvolution(48, 24, 5, 5, 1, 1, 2, 2)(encoder_mp2)
  local mp3 = nn.SpatialMaxPooling(2, 2)
  mp3_shell = nn.Sequential()
  mp3_shell:add(mp3)
  encoder_mp3 = mp3_shell(encoder_mp2)
  encoder_mp3 = nn.Threshold(0, 1e-6)(encoder_mp3)

  encoder_mp3 = nn.Reshape(24*8*8)(encoder_mp3)

  encoder = nn.Linear(24*8*8, opt.z_dim)(encoder_mp3)
  
    -- the z code
  local z = nn.Sigmoid()(encoder)

    -- code for A
  encoder_A = nn.Identity()(z)
  encoder_A = nn.Linear(opt.z_dim, opt.zA_dim)(encoder_A)
  local zA = nn.Identity()(encoder_A)

    -- code for N
  encoder_N = nn.Identity()(z)
  encoder_N = nn.Linear(opt.z_dim, opt.zN_dim)(encoder_N)
  local zN = nn.Identity()(encoder_N)

    -- code for L
  encoder_L = nn.Identity()(z)  
  local zL = nn.Linear(opt.z_dim, opt.zL_dim)(encoder_L)
  
    -- code for B
  encoder_B = nn.Identity()(z)
  local zB = nn.Linear(opt.z_dim, opt.zB_dim)(encoder_B)

    -- code for M
  encoder_M = nn.Identity()(z)
  local zM = nn.Linear(opt.z_dim, opt.zM_dim)(encoder_M)

  ----------------------------------------------------------------
      -- the express decoder(s) for B (background) and M (matte)
  -------------------------------------------------------
        -- the decoder for B
  zB2 = nn.Identity()()      
  decoder_B = nn.Identity()(zB2)
  decoder_B = nn.Linear(opt.zB_dim, 24*8*8)(decoder_B)
  decoder_B = nn.Threshold(0, 1e-6)(decoder_B)
  decoder_B = nn.Reshape(24, 8, 8)(decoder_B)
  
  decoder_B = nn.SpatialMaxUnpooling(mp3)(decoder_B)
  decoder_B = cudnn.SpatialConvolution(24, 48, 5, 5, 1, 1, 2, 2)(decoder_B)
  decoder_B = nn.Threshold(0, 1e-6)(decoder_B)

  decoder_B = nn.SpatialMaxUnpooling(mp2)(decoder_B)
  decoder_B = cudnn.SpatialConvolution(48, 96, 5, 5, 1, 1, 2, 2)(decoder_B)
  decoder_B = nn.Threshold(0, 1e-6)(decoder_B)

  decoder_B = nn.SpatialMaxUnpooling(mp1)(decoder_B)
  decoder_B = cudnn.SpatialConvolution(96, 96, 5, 5, 1, 1, 2, 2)(decoder_B)
  decoder_B = nn.Threshold(0, 1e-6)(decoder_B)

  decoder_B = cudnn.SpatialConvolution(96, 3, 3, 3, 1, 1, 1, 1)(decoder_B) 
  local B = nn.HardTanh()(decoder_B)

          -- the decoder for M
  zM2 = nn.Identity()()      
  decoder_M = nn.Identity()(zM2)
  decoder_M = nn.Linear(opt.zM_dim, 24*8*8)(decoder_M)
  decoder_M = nn.Threshold(0, 1e-6)(decoder_M)
  decoder_M = nn.Reshape(24, 8, 8)(decoder_M)
  
  decoder_M = nn.SpatialMaxUnpooling(mp3)(decoder_M)
  decoder_M = cudnn.SpatialConvolution(24, 48, 5, 5, 1, 1, 2, 2)(decoder_M)
  decoder_M = nn.Threshold(0, 1e-6)(decoder_M)

  decoder_M = nn.SpatialMaxUnpooling(mp2)(decoder_M)
  decoder_M = cudnn.SpatialConvolution(48, 96, 5, 5, 1, 1, 2, 2)(decoder_M)
  decoder_M = nn.Threshold(0, 1e-6)(decoder_M)

  decoder_M = nn.SpatialMaxUnpooling(mp1)(decoder_M)
  decoder_M = cudnn.SpatialConvolution(96, 96, 5, 5, 1, 1, 2, 2)(decoder_M)
  decoder_M = nn.Threshold(0, 1e-6)(decoder_M)

  decoder_M = cudnn.SpatialConvolution(96, 3, 3, 3, 1, 1, 1, 1)(decoder_M) 
  local M = nn.HardTanh()(decoder_M)


  ----------------------------------------------------------------
      -- the decoder(s)
  -------------------------------------------------------
        -- the decoder for A

  zA2 = nn.Identity()()      
  decoder_A = nn.Identity()(zA2)
  decoder_A = nn.Linear(opt.zA_dim, 24*8*8)(decoder_A)
  decoder_A = nn.Threshold(0, 1e-6)(decoder_A)
  decoder_A = nn.Reshape(24, 8, 8)(decoder_A)

  decoder_A = nn.SpatialUpSamplingNearest(2)(decoder_A)
  decoder_A = cudnn.SpatialConvolution(24, 48, 5, 5, 1, 1, 2, 2)(decoder_A)
  decoder_A = nn.Threshold(0, 1e-6)(decoder_A)

  decoder_A = nn.SpatialUpSamplingNearest(2)(decoder_A)
  decoder_A = cudnn.SpatialConvolution(48, 96, 5, 5, 1, 1, 2, 2)(decoder_A)
  decoder_A = nn.Threshold(0, 1e-6)(decoder_A)

  decoder_A = nn.SpatialUpSamplingNearest(2)(decoder_A)
  decoder_A = cudnn.SpatialConvolution(96, 96, 5, 5, 1, 1, 2, 2)(decoder_A)
  decoder_A = nn.Threshold(0, 1e-6)(decoder_A)

  decoder_A = cudnn.SpatialConvolution(96, 3, 3, 3, 1, 1, 1, 1)(decoder_A)
  
  local A = nn.Threshold(0, 1e-6)(decoder_A)

  -------------------------------------------------------
        -- the decoder for N
  zN2 = nn.Identity()()      
  decoder_N = nn.Identity()(zN2)
  decoder_N = nn.Linear(opt.zN_dim, 24*8*8)(decoder_N)
  decoder_N = nn.Tanh()(decoder_N)
  decoder_N = nn.Reshape(24, 8, 8)(decoder_N)
  decoder_N = nn.SpatialUpSamplingNearest(2)(decoder_N)
  decoder_N = cudnn.SpatialConvolution(24, 48, 5, 5, 1, 1, 2, 2)(decoder_N)
  decoder_N = nn.Tanh()(decoder_N)
  decoder_N = nn.SpatialUpSamplingNearest(2)(decoder_N)
  decoder_N = cudnn.SpatialConvolution(48, 96, 5, 5, 1, 1, 2, 2)(decoder_N)
  decoder_N = nn.Tanh()(decoder_N)
  decoder_N = nn.SpatialUpSamplingNearest(2)(decoder_N)
  decoder_N = cudnn.SpatialConvolution(96, 96, 5, 5, 1, 1, 2, 2)(decoder_N)
  decoder_N = nn.Tanh()(decoder_N)
  decoder_N = cudnn.SpatialConvolution(96, 2, 3, 3, 1, 1, 1, 1)(decoder_N)


  N_at = nn.Identity()()
  N_at = nn.Identity()(decoder_N)

  Nproc = nn.Sequential()
  Nproc:add(nn.SplitTable(1,3))
  Nxp, Nyp = Nproc(N_at):split(2)
  
  Nxp = nn.HardTanh()(Nxp)      -- Nx: [-1, 1]
  Nxp = nn.View(1, opt.scale, opt.scale)(Nxp)
  Nyp = nn.HardTanh()(Nyp)      -- Ny: [-1, 1]
  Nyp = nn.View(1, opt.scale, opt.scale)(Nyp)

  -- compute Nz from Nz and Ny
  Nxpsq = nn.Square()(Nxp)
  Nypsq = nn.Square()(Nyp)
  Nzpsq = nn.CAddTable()({Nxpsq,Nypsq})
  Nzpsq = nn.AddConstant(-1)(Nzpsq)
  Nzpsq = nn.MulConstant(-1)(Nzpsq)
  Nzpsq = nn.ReLU()(Nzpsq) 
  Nzp = nn.Sqrt()(Nzpsq)    
  --Nzp = nn.HardTanh()(Nzp)    -- Nz: [0, 1]
  Nzp = nn.View(1, opt.scale, opt.scale)(Nzp)

  local N = nn.JoinTable(1,3)({Nxp,Nyp,Nzp})
  -- The norm(squared of normal map)
  local Nnm = nn.PerElementNorm()(N)
  -------------------------------------------------------
        -- the decoder for Ls : Lr, Lg and Lb
  zL2 = nn.Identity()()
  decoder_L = nn.Identity()(zL2)
  local Lr = nn.Linear(opt.zL_dim , 10)(decoder_L)
  local Lg = nn.Linear(opt.zL_dim , 10)(decoder_L)
  local Lb = nn.Linear(opt.zL_dim , 10)(decoder_L)

  -------------------------------------------------------
  -- wrap it up
  mask = nn.Identity()()

  generate_S = nn.SHPartialShadingRGB_bw()({N, Lr, Lg, Lb, mask})

  local S = cudnn.ReLU()(generate_S) -- ReLu the shading

  --local S2 = nn.ShadingMask()({S,mask,bias}) -- mask the shading
  -- compose image
  generate_I = nn.SACompose()({A,S})
  local synth = nn.HardTanh(0,1)(generate_I)

  -- model_G = nn.gModule({e_input, mask},{synth, A, N, L, S})

  generate_final = nn.FBMCompose()({synth,B,M})
  local final = nn.HardTanh(0,1)(generate_final)

  -- encoder model
  model_Enc = nn.gModule({e_input}, {zA, zN, zL, zB, zM})
  -- decoder model
  model_Dec = nn.gModule({zA2, zN2, zL2, zB2, zM2, mask},{synth, A, N, Lr, Lg, Lb, S, Nnm, B, M, final})
  
  ----------------------------------------------------------------------

else
  print('<trainer> reloading previously trained network: ' .. opt.network)
  tmp = torch.load(opt.network)
  model_D = tmp.D
  -- model_G = tmp.G
  model_Enc = tmp.Enc
  model_Dec = tmp.Dec
end

------------------------------------------
-- loss function: negative log-likelihood
------------------------------------------
criterion_ebadv_real = nn.MSECriterion()
criterion_ebadv_gene = nn.MarginNegMSECriterion()

criterion_ebadv_G = nn.MSECriterion()

criterion_ebadv_test = nn.MSECriterion()

criterion_BCE = nn.BCECriterion()
criterion_adv = nn.BCECriterion()
criterion_rec = nn.MSECriterion() 
criterion_abs = nn.AbsCriterion()
criterion_N = nn.MSECriterion()
criterion_A = nn.MSECriterion()
criterion_Lr = nn.LightCoeffCriterion()
criterion_Lg = nn.LightCoeffCriterion()
criterion_Lb = nn.LightCoeffCriterion()
criterion_S = nn.RangeSelfCriterion()
criterion_A_tv = nn.TVSelfCriterion()
criterion_A_tv_partial = nn.TVSelfPartialCriterion()
criterion_A_range = nn.RangeSelfCriterion()
criterion_N_smooth = nn.SmoothSelfCriterion()
criterion_N_smooth_partial = nn.SmoothSelfPartialCriterion()
criterion_S_smooth = nn.SmoothSelfCriterion()
criterion_N_sim_partial = nn.PartialSimNCriterion()
criterion_Nnm = nn.MSECriterion()

criterion_S_bw = nn.BatchWhiteShadingCriterion()
criterion_S_bw2 = nn.BatchWhiteShadingCriterion2()

criterion_maskrecon = nn.MaskedReconCriterion()
criterion_B = nn.MSECriterion()
criterion_M = nn.AbsCriterion()
criterion_M_smooth =  nn.SmoothSelfCriterion()

criterion_final_abs = nn.AbsCriterion() 

-------------------------------------
-- retrieve parameters and gradients
-------------------------------------
parameters_D,gradParameters_D = model_D:getParameters()
parameters_Enc,gradParameters_Enc = model_Enc:getParameters()
parameters_Dec,gradParameters_Dec = model_Dec:getParameters()

-- print networks
print('Discriminator network:')
print(model_D)
print('Encoder network:')
print(model_Enc)
print('Decoder network:')
print(model_Dec)

-- this matrix records the current confusion across classes
classes = {'0','1'}
confusion = optim.ConfusionMatrix(classes)

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))


if opt.gpu then
  print('Copy model to gpu')
  model_D:cuda()
  model_Enc:cuda()
  model_Dec:cuda()
end

-- Training parameters
sgdState_D = {
  learningRate = opt.learningRate,
  momentum = opt.momentum,
  optimize=true,
  numUpdates = 0
}

sgdState_G = {
  learningRate = opt.learningRate,
  momentum = opt.momentum,
  optimize=true,
  numUpdates=0
}

sgdState_Enc = {
  learningRate = opt.learningRate,
  momentum = opt.momentum,
  optimize=true,
  numUpdates=0
}
sgdState_Dec = {
  learningRate = opt.learningRate,
  momentum = opt.momentum,
  optimize=true,
  numUpdates=0
}

----------------------------------------------------------------------
-- Get examples to plot
function getSamples(dataset, N)
  local numperclass = numperclass or 10
  local N = N or 8
  --local noise_inputs = torch.Tensor(N, opt.noiseDim)

  -- Generate samples
  local inputs_samples_img = torch.Tensor(N, opt.geometry[1], opt.geometry[2], opt.geometry[3])
  local inputs_samples_mask = torch.Tensor(N, opt.geometry[1], opt.geometry[2], opt.geometry[3])
  --local inputs_samples_bias = torch.Tensor(N, opt.geometry[1], opt.geometry[2], opt.geometry[3])
  math.randomseed( os.time() )
  for i = 1,N do
    local idx = math.random(dataset:size()[1])
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
  local zA_sample, zN_sample, zL_sample, zB_sample, zM_sample = unpack(model_Enc:forward(inputs_samples_img))
  local synth_samples,A_samples,N_samples,Lr_samples,Lg_samples,Lb_samples,S_samples, Nnm_samples, B_samples, M_samples, final_samples = unpack(model_Dec:forward({zA_sample, zN_sample, zL_sample, zB_sample, zM_sample, inputs_samples_mask}))
  --local synth_samples,A_samples,N_samples,L_samples,S_samples, M_samples = unpack(model_G:forward({inputs_samples_img, inputs_samples_mask}))
  print(synth_samples:size())
  synth_samples = nn.HardTanh():forward(synth_samples)
  A_samples = nn.HardTanh():forward(A_samples)
  N_samples = nn.HardTanh():forward(N_samples)
  S_samples = nn.Tanh():forward(S_samples)
  N_samples = (N_samples+1)*0.5
  B_samples = nn.HardTanh():forward(B_samples)
  M_samples = nn.HardTanh():forward(M_samples)
  final_samples = nn.HardTanh():forward(final_samples)
  --N_samples = 0.5*(N_samples+1)
  --local samplesplit
  --samplesplit = samples:chunk(2,2)  -- split the 6-d image to two 3-d images
  --samples_1 = samplesplit[1]
  --samples_2 = samplesplit[2]

  local to_plot_0 = {}
  local to_plot_1 = {}
  local to_plot_2 = {}
  local to_plot_3 = {}
  local to_plot_4 = {}
  local to_plot_5 = {}
  local to_plot_6 = {}
  local to_plot_7 = {}
  for i=1,N do
    to_plot_0[#to_plot_0 + 1] = inputs_samples_img[i]:float()
    to_plot_1[#to_plot_1 + 1] = synth_samples[i]:float()
    to_plot_2[#to_plot_2 + 1] = A_samples[i]:float()
    to_plot_3[#to_plot_3 + 1] = N_samples[i]:float()
    to_plot_4[#to_plot_4 + 1] = S_samples[i]:float()
    to_plot_5[#to_plot_5 + 1] = B_samples[i]:float()
    to_plot_6[#to_plot_6 + 1] = M_samples[i]:float()
    to_plot_7[#to_plot_7 + 1] = final_samples[i]:float()
  end

  return to_plot_0, to_plot_1, to_plot_2, to_plot_3, to_plot_4 ,to_plot_5, to_plot_6, to_plot_7
end

----------------------------------------------------
------------ data loading and training -------------
----------------------------------------------------
-- load data, need to write a script for more flexible data loading 
-- training set table, get training data in the loop
trainSets = {}
trainSets_light = {}
trainSets[1] = 'data/zx_7_d10_inmc_celebA_00.hdf5'
trainSets[2] = 'data/zx_7_d10_inmc_celebA_01.hdf5'
trainSets[3] = 'data/zx_7_d10_inmc_celebA_02.hdf5'
trainSets[4] = 'data/zx_7_d10_inmc_celebA_03.hdf5'
trainSets[5] = 'data/zx_7_d10_inmc_celebA_04.hdf5'
trainSets[6] = 'data/zx_7_d10_inmc_celebA_05.hdf5'
trainSets[7] = 'data/zx_7_d10_inmc_celebA_06.hdf5'
trainSets[8] = 'data/zx_7_d10_inmc_celebA_07.hdf5'
trainSets[9] = 'data/zx_7_d10_inmc_celebA_08.hdf5'
trainSets[10] = 'data/zx_7_d10_inmc_celebA_09.hdf5'
trainSets[11] = 'data/zx_7_d10_inmc_celebA_10.hdf5'
trainSets[12] = 'data/zx_7_d10_inmc_celebA_11.hdf5'
trainSets[13] = 'data/zx_7_d10_inmc_celebA_12.hdf5'
trainSets[14] = 'data/zx_7_d10_inmc_celebA_13.hdf5'
trainSets[15] = 'data/zx_7_d10_inmc_celebA_14.hdf5'
trainSets[16] = 'data/zx_7_d10_inmc_celebA_15.hdf5'
trainSets[17] = 'data/zx_7_d10_inmc_celebA_16.hdf5'
trainSets[18] = 'data/zx_7_d10_inmc_celebA_17.hdf5'
trainSets[19] = 'data/zx_7_d10_inmc_celebA_18.hdf5'
trainSets[20] = 'data/zx_7_d10_inmc_celebA_19.hdf5'
trainSets_light[1] = 'data/zx_7_d3_lrgb_celebA_00.hdf5'
trainSets_light[2] = 'data/zx_7_d3_lrgb_celebA_01.hdf5'
trainSets_light[3] = 'data/zx_7_d3_lrgb_celebA_02.hdf5'
trainSets_light[4] = 'data/zx_7_d3_lrgb_celebA_03.hdf5'
trainSets_light[5] = 'data/zx_7_d3_lrgb_celebA_04.hdf5'
trainSets_light[6] = 'data/zx_7_d3_lrgb_celebA_05.hdf5'
trainSets_light[7] = 'data/zx_7_d3_lrgb_celebA_06.hdf5'
trainSets_light[8] = 'data/zx_7_d3_lrgb_celebA_07.hdf5'
trainSets_light[9] = 'data/zx_7_d3_lrgb_celebA_08.hdf5'
trainSets_light[10] = 'data/zx_7_d3_lrgb_celebA_09.hdf5'
trainSets_light[11] = 'data/zx_7_d3_lrgb_celebA_10.hdf5'
trainSets_light[12] = 'data/zx_7_d3_lrgb_celebA_11.hdf5'
trainSets_light[13] = 'data/zx_7_d3_lrgb_celebA_12.hdf5'
trainSets_light[14] = 'data/zx_7_d3_lrgb_celebA_13.hdf5'
trainSets_light[15] = 'data/zx_7_d3_lrgb_celebA_14.hdf5'
trainSets_light[16] = 'data/zx_7_d3_lrgb_celebA_15.hdf5'
trainSets_light[17] = 'data/zx_7_d3_lrgb_celebA_16.hdf5'
trainSets_light[18] = 'data/zx_7_d3_lrgb_celebA_17.hdf5'
trainSets_light[19] = 'data/zx_7_d3_lrgb_celebA_18.hdf5'
trainSets_light[20] = 'data/zx_7_d3_lrgb_celebA_19.hdf5'
nTrainSet = 20
-- get validation data
local lfwHd5 = hdf5.open('data/zx_7_d10_inmc_celebA_20.hdf5', 'r')
valData = lfwHd5:read('zx_7'):all()
-- data:mul(2):add(-1) -- convert from [0,1] to [-1, 1]
lfwHd5:close()

-- training loop
while true do
  print("One epoch started .. ")
--while true do
  print("Sample images .. ")
  local to_plot_0, to_plot_1, to_plot_2, to_plot_3, to_plot_4, to_plot_5, to_plot_6, to_plot_7 = getSamples(valData, 49)
  print("Finished sample images .. ")

  torch.setdefaulttensortype('torch.FloatTensor')

  print("Save sampled images .. ")
  local formatted_0 = image.toDisplayTensor({input=to_plot_0, nrow=7})
  formatted_0:float()
  image.save(opt.save .."/example_origin_"..(epoch or 0)..'.png', formatted_0)

  local formatted_1 = image.toDisplayTensor({input=to_plot_1, nrow=7})
  formatted_1:float()
  image.save(opt.save .."/example_syn_"..(epoch or 0)..'.png', formatted_1)

  local formatted_2 = image.toDisplayTensor({input=to_plot_2, nrow=7})
  formatted_2:float()
  image.save(opt.save .."/example_A_"..(epoch or 0)..'.png', formatted_2)

  local formatted_3 = image.toDisplayTensor({input=to_plot_3, nrow=7})
  formatted_3:float()
  image.save(opt.save .."/example_N_"..(epoch or 0)..'.png', formatted_3)

  local formatted_4 = image.toDisplayTensor({input=to_plot_4, nrow=7})
  formatted_4:float()
  image.save(opt.save .."/example_S_"..(epoch or 0)..'.png', formatted_4)

  local formatted_5 = image.toDisplayTensor({input=to_plot_5, nrow=7})
  formatted_5:float()
  image.save(opt.save .."/example_B_"..(epoch or 0)..'.png', formatted_5)

  local formatted_6 = image.toDisplayTensor({input=to_plot_6, nrow=7})
  formatted_6:float()
  image.save(opt.save .."/example_M_"..(epoch or 0)..'.png', formatted_6)

  local formatted_7 = image.toDisplayTensor({input=to_plot_7, nrow=7})
  formatted_7:float()
  image.save(opt.save .."/example_final_"..(epoch or 0)..'.png', formatted_7)
  -- image.save(opt.save .."/ridiculous"..(epoch or 0)..'.png', abcd)

  if opt.gpu then
    torch.setdefaulttensortype('torch.CudaTensor')
  else
    torch.setdefaulttensortype('torch.FloatTensor')
  end

  -- train/test
  train_idx = math.random(nTrainSet)
  print("Working on subset: ", trainSets[train_idx]) 
  print("Open data file .. ")
  local dataHd5 = hdf5.open(trainSets[train_idx], 'r')
  trainData = dataHd5:read('zx_7'):all()
  dataHd5:close()
  print("Data file closed .. ")
  print("Open light file .. ")
  local lightHd5 = hdf5.open(trainSets_light[train_idx], 'r')
  trainLight = lightHd5:read('zx_7'):all()
  lightHd5:close()
  print("Light file closed .. ")
  formation.train(trainData, trainLight)
  
  -- local debugger = require('fb.debugger')
  -- debugger.enter()
 
  formation.test(valData)

  sgdState_D.momentum = math.min(sgdState_D.momentum + 0.0008, 0.7)
  sgdState_D.learningRate = math.max(opt.learningRate*0.99^epoch, 0.000001)
  
  sgdState_Enc.momentum = math.min(sgdState_Enc.momentum + 0.0008, 0.7)
  sgdState_Enc.learningRate = math.max(opt.learningRate*0.99^epoch, 0.000001)
  
  sgdState_Dec.momentum = math.min(sgdState_Dec.momentum + 0.0008, 0.7)
  sgdState_Dec.learningRate = math.max(opt.learningRate*0.99^epoch, 0.000001)

  print("One epoch finished .. ")
  
end


