------------------------------------------------------------
--- Code for 
-- Shu et al., Neural Face Editing with Intrinsic Image Disentangling, CVPR 2017.
------------------------------------------------------------
require 'torch'
require 'nn'
require 'cunn'
require 'optim'
require 'pl'
require 'image'


local formation = {}

local function sleep(n)
  os.execute("sleep " .. tonumber(n))
end

--
--
function optim.rmsprop(opfunc, x, config, state)
    -- this function will update x (a.k.a model parameters) with optim.rmsprop
    -- the gradient is compute via a function opfunc (a handle of the function passed to the function)
    -- (0) get/update state
    local config = config or {}
    local state = state or config
    local lr = config.learningRate or 1e-2
    local alpha = config.alpha or 0.9
    local epsilon = config.epsilon or 1e-8

    -- (1) evaluate f(x) and df/dx
    -- what was passed was the handle of the function
    -- x is the parameters, what is returned is obejctive fx and the gradient of parameters gradParameters
    local fx, dfdx = opfunc(x)
    if config.optimize == true then
        -- (2) initialize mean square values and square gradient storage
        if not state.m then
          state.m = torch.Tensor():typeAs(x):resizeAs(dfdx):zero()
          state.tmp = torch.Tensor():typeAs(x):resizeAs(dfdx)
        end

        -- (3) calculate new (leaky) mean squared values
        state.m:mul(alpha)
        state.m:addcmul(1.0-alpha, dfdx, dfdx)

        -- (4) perform update
        state.tmp:sqrt(state.m):add(epsilon)
        -- only opdate when optimize is true
        
        
  if config.numUpdates < 10 then
        io.write(" ", lr/50.0, " ")
        x:addcdiv(-lr/50.0, dfdx, state.tmp)
  elseif config.numUpdates < 30 then
      io.write(" ", lr/5.0, " ")
      x:addcdiv(-lr /5.0, dfdx, state.tmp)
  else 
    io.write(" ", lr, " ")
    x:addcdiv(-lr, dfdx, state.tmp)
  end
    end
    config.numUpdates = config.numUpdates +1
    -- return x*, f(x) before optimization
    return x, {fx}
end

--
--
--

function adam(opfunc, x, config, state)
    -- this function will update x (a.k.a model parameters) with ADAM
    -- the gradient is compute via a function opfunc (a handle of the function passed to the function)
    --print('ADAM')
    -- (0) get/update state
    local config = config or {}
    local state = state or config
    local lr = config.learningRate or 0.001

    local beta1 = config.beta1 or 0.9
    local beta2 = config.beta2 or 0.999
    local epsilon = config.epsilon or 1e-8

    -- (1) evaluate f(x) and df/dx
    local fx, dfdx = opfunc(x)
    if config.optimize == true then
      -- Initialization
      state.t = state.t or 0
      -- Exponential moving average of gradient values
      state.m = state.m or x.new(dfdx:size()):zero()
      -- Exponential moving average of squared gradient values
      state.v = state.v or x.new(dfdx:size()):zero()
      -- A tmp tensor to hold the sqrt(v) + epsilon
      state.denom = state.denom or x.new(dfdx:size()):zero()

      state.t = state.t + 1
      
      -- Decay the first and second moment running average coefficient
      state.m:mul(beta1):add(1-beta1, dfdx)
      state.v:mul(beta2):addcmul(1-beta2, dfdx, dfdx)

      state.denom:copy(state.v):sqrt():add(epsilon)

      local biasCorrection1 = 1 - beta1^state.t
      local biasCorrection2 = 1 - beta2^state.t
      
    local fac = 1
    if config.numUpdates < 10 then
        fac = 50.0
    elseif config.numUpdates < 30 then
        fac = 5.0
    else 
        fac = 1.0
    end
    io.write(" ", lr/fac, " ")
        local stepSize = (lr/fac) * math.sqrt(biasCorrection2)/biasCorrection1
      -- (2) update x
      x:addcdiv(-stepSize, state.m, state.denom)
    end
    config.numUpdates = config.numUpdates +1
    -- return x*, f(x) before optimization
    return x, {fx}
end

--
-- training function
--
function formation.train(dataset, data_add, N)
  -- set the Encoder and Decoder model to training mode, basically enable functionalities like dropout etc
  model_Enc:training()
  model_Dec:training()
  -- set the D model to training mode, basically enbale functionalities like dropout etc
  model_D:training()


  epoch = epoch or 1
  local N = N or dataset:size()[1]
  -- dataBatchSize is half of the batchSize, the other half are noise
  local dataBatchSize = opt.batchSize / 2
  local time = sys.clock()

  -- do one epoch
  print('\n<trainer> on training set:')
  print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ' lr = ' .. sgdState_D.learningRate .. ', momentum = ' .. sgdState_D.momentum .. ']')
  for t = 1,N,dataBatchSize do

    -- opt.geometry is the size of the input sample, e.g. 3*64*64
    -- these are the inputs of the D net, includes half real and half G net output
    -- basically first half will be direct sampling from dataset, second half will be reconstruction via G net from some other input
    local inputs = torch.Tensor(opt.batchSize, opt.geometry[1], opt.geometry[2], opt.geometry[3])
    -- these are the inputs of the G net, half size of the D net input
    local g_inputs_img = torch.Tensor(dataBatchSize, opt.geometry[1], opt.geometry[2], opt.geometry[3])
    local g_inputs_mask = torch.Tensor(dataBatchSize, opt.geometry[1], opt.geometry[2], opt.geometry[3])
    --local g_inputs_bias = torch.Tensor(dataBatchSize, opt.geometry[1], opt.geometry[2], opt.geometry[3])
    local g_outputs = torch.Tensor(dataBatchSize, opt.geometry[1], opt.geometry[2], opt.geometry[3])    
    -- target , what is it? this is the corresponding ideal/target/ground truth ouput of the input of the D network
    local targets = torch.Tensor(opt.batchSize)
    -- inputs for the training of G net
    local g_inputs2_img = torch.Tensor(opt.batchSize, opt.geometry[1], opt.geometry[2], opt.geometry[3])
    local g_inputs2_geo = torch.Tensor(opt.batchSize, opt.geometry[1], opt.geometry[2], opt.geometry[3])
    local g_inputs2_mask = torch.Tensor(opt.batchSize, opt.geometry[1], opt.geometry[2], opt.geometry[3])
    local g_inputs2_Lr = torch.Tensor(opt.batchSize, 9)
    local g_inputs2_Lg = torch.Tensor(opt.batchSize, 9)
    local g_inputs2_Lb = torch.Tensor(opt.batchSize, 9)
    ----------------------------------------------------------------------
    -- create closure to evaluate f(X) and df/dX of discriminator
    -- handle of the function is called in the optimizer
    -- the data consists of inputs and targets
    ----------------------------------------------------------------------    
    local fevalD = function(x)
      collectgarbage()
      if x ~= parameters_D then -- get new parameters
        parameters_D:copy(x)
      end

      gradParameters_D:zero() -- reset gradients

      -- forward pass on the D network
      -- training data: inputs, generated in the later part of the code (while training)
      -- inputs are just face images, generated ones and real ones
      -- real inputs
      local inputs_real = inputs[{{1,dataBatchSize},{},{},{}}]:clone()
      -- generation inputs
      local inputs_gene = inputs[{{dataBatchSize+1,opt.batchSize},{},{},{}}]:clone()

      local outputs = model_D:forward(inputs)
      local outputs_real = outputs[{{1,dataBatchSize},{},{},{}}]:clone()
      local outputs_gene = outputs[{{dataBatchSize+1,opt.batchSize},{},{},{}}]:clone()
      
      -- objective and gradient for D-enc-dec network
      local f_real = criterion_ebadv_real:forward(inputs_real, outputs_real)
      local df_real = criterion_ebadv_real:backward(inputs_real, outputs_real)

      local f_gene = criterion_ebadv_gene:forward(inputs_gene, outputs_gene)
      local df_gene = criterion_ebadv_gene:backward(inputs_gene, outputs_gene)



      sgdState_D.optimize = true
      sgdState_Dec.optimize = true
      sgdState_Enc.optimize = true

  
      --print(monA:size(), tarA:size())
      io.write("v1_lfw| R:", f_real,"  F:", f_gene, "  ")
      -- error/objective in general
      local f = f_real*1 + f_gene*1

      df_real = df_real*1
      df_gene = df_gene*1
      local df_do = torch.cat(df_real, df_gene, 1)

      -- feedback to previous layers
      model_D:backward(inputs, df_do)

      -- penalties (L1 and L2):
      if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
        local norm,sign= torch.norm,torch.sign
        -- Loss:
          -- L1 regularization
        f = f + opt.coefL1 * norm(parameters_D,1)
          -- L2 regularization
        f = f + opt.coefL2 * norm(parameters_D,2)^2/2
        -- Gradients:
        gradParameters_D:add( sign(parameters_D):mul(opt.coefL1) + parameters_D:clone():mul(opt.coefL2) )
      end
      return f,gradParameters_D
    end

    ----------------------------------------------------------------------
    -- create closure to evaluate f(X) and df/dX of decoder Dec 
    ----------------------------------------------------------------------
    local fevalDec = function(x)
      collectgarbage()
      if x ~= parameters_Dec then -- get new parameters
        parameters_Dec:copy(x)
      end
      
      gradParameters_Dec:zero() -- reset gradients

      local f_all = f_synth + f_A + f_N + f_Lr + f_Lg + f_Lb + f_S  + f_Nnm + f_B + f_M + f_final

      io.write("Dec:",f_all, " Dec:", tostring(sgdState_Dec.optimize)," D:",tostring(sgdState_D.optimize)," ", sgdState_Dec.numUpdates, " ", sgdState_D.numUpdates , "\n")
      io.flush()

      local df_all = {df_synth, df_A, df_N, df_Lr, df_Lg, df_Lb, df_S, df_Nnm, df_B, df_M, df_final}
      -- backwards the error to the decoder input
      df_zA, df_zN, df_zL, df_zB, df_zM = unpack(model_Dec:backward(dec_input, df_all))

      print('gradParameters_Dec', gradParameters_Dec:norm())

      return f_all, gradParameters_Dec
    end

    ----------------------------------------------------------------------
    -- create closure to evaluate f(X) and df/dX of encoder Enc
    ----------------------------------------------------------------------
    local fevalEnc = function(x)
      collectgarbage()
      if x ~= parameters_Enc then -- get new parameters
        parameters_Enc:copy(x)
      end
      
      gradParameters_Enc:zero() -- reset gradients

      local f_all = f_synth + f_A + f_N +  f_Lr + f_Lg + f_Lb  + f_S + f_Nnm + f_B + f_M + f_final
      
      model_Enc:backward( enc_input , {df_zA, df_zN, df_zL, df_zB, df_zM})
      
      io.write("Enc:",f_all, " Enc:", tostring(sgdState_Enc.optimize)," D:",tostring(sgdState_D.optimize)," ", sgdState_Enc.numUpdates, " ", sgdState_D.numUpdates , "\n")
      io.flush()
      print('gradParameters_Enc', gradParameters_Enc:norm())

      return f_all, gradParameters_Enc
    end

    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z))), which means, a classification objective
    -- Get half a minibatch of real, half fake
    -- K is the number of iterations to optimize D-net, use 1 as default
    for k=1,opt.K do
      -- (1.1) Real data 
      local id = 1
      local pivot = math.min(t+dataBatchSize-1,dataset:size()[1])
      local lnew = pivot - t 
      -- create real data input from training set
      math.randomseed( os.time() )
      for i = t,pivot do
            -- sample: those samples that are directly feed into D net
        local idx = math.random(dataset:size()[1])
        local sample = dataset[idx] 
        local sample_img = sample:narrow(1,1,3) -- select first three dimensions of the sample
            -- sample2: those samples that are going through the reconstruction before going into D net
        local idx2 = math.random(dataset:size()[1])
        local sample2 = dataset[idx2]
        local sample2_img =  sample2:narrow(1,1,3) -- select first three dimensions of the sample
        local sample2_mask = sample2:narrow(1,7,1)
        --local sample2_bias = sample2_img:clone()
        sample2_mask = torch.repeatTensor(sample2_mask,3,1,1)
        --sample2_bias = sample2_bias:fill(1)

        -- inputs is not delcared as local, therefore it's default global
        inputs[id] = sample_img:clone()
        g_inputs_img[id] = sample2_img:clone()
        g_inputs_mask[id] = sample2_mask:clone()
        --g_inputs_bias[id] = sample2_bias:clone()
        id = id + 1
      end
      -- the target of the real data input, of course, positive labeling (1)
      -- targets is not declared as local, therefore it's default global
      targets[{{1,dataBatchSize}}]:fill(1)

      -- (1.2) Sampled data
      -- noise_inputs:normal(0, 1)
      -- create fake data input from G-net's output
      id = id + dataBatchSize - lnew -1 
      local sample_g_zA, sample_g_zN, sample_g_zL, sample_g_zB, sample_g_zM = unpack(model_Enc:forward(g_inputs_img))
      local g_samples_synth, g_samples_A, g_samples_N, g_samples_Lr, g_samples_Lg, g_samples_Lb, g_samples_S, g_samples_Nnm, g_samples_B, g_samples_M, g_samples_img = unpack(model_Dec:forward({sample_g_zA, sample_g_zN, sample_g_zL, sample_g_zB, sample_g_zM, g_inputs_mask}))
      --local g_samples_img = g_samples[1] -- samples_all contains synth, A, N, L. But we only need synth
      for i = 1,dataBatchSize do
        inputs[id] = g_samples_img[i]:clone()
        id = id + 1
      end
      -- the target of the fake data input, which is negative labeling (-1)
      targets[{{dataBatchSize+1,opt.batchSize}}]:fill(0)

      optim.rmsprop(fevalD, parameters_D, sgdState_D)

    end -- end for K

    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    -- (2) Update Enc/Dec network: 
      -- prepare batch data for training
    math.randomseed( os.time() )
    for i = 1,opt.batchSize do
      local idx = math.random(dataset:size()[1])
      local sample3 = dataset[idx]
      local sample3_img = sample3:narrow(1,1,3)
      local sample3_geo = sample3:narrow(1,4,3)
      local sample3_mask = sample3:narrow(1,7,1)
      local sample3_light = data_add[idx]
      --local sample3_bias = sample3_img:clone()
      sample3_geo = sample3_geo*2 - 1 -- range from [0,1] to [-1, 1]
      sample3_mask = torch.repeatTensor(sample3_mask,3,1,1)
      g_inputs2_img[i] = sample3_img:clone()
      g_inputs2_geo[i] = sample3_geo:clone()
      g_inputs2_mask[i] = sample3_mask:clone()
      g_inputs2_Lr[i] = sample3_light[1]:clone()
      g_inputs2_Lg[i] = sample3_light[2]:clone()
      g_inputs2_Lb[i] = sample3_light[3]:clone()
    end    
    
-- (2.1) adverserial loss
      -- maximize log(D(G(z))), adverserail loss: in training, put all target label for ``fake'' input to 1
       -- initialize g_inputs
    targets:fill(1)
    
      -- forward pass: Enc, from input image to code z
    enc_input = g_inputs2_img -- input of the decoder
    g_zA, g_zN, g_zL, g_zB, g_zM = unpack(model_Enc:forward(enc_input))
    
      -- forward pass: Dec, from code z (and mask) to reconstruction and all the other layers
    dec_input = {g_zA, g_zN, g_zL, g_zB, g_zM, g_inputs2_mask} -- input of the decoder
    g_synth, g_A, g_N, g_Lr, g_Lg, g_Lb, g_S, g_Nnm, g_B, g_M, g_final = unpack(model_Dec:forward(dec_input))

    -- formation loss from D net
      -- for adv loss, forward g_synth to the output layer
    output_adv = model_D:forward(g_final)
    f_adv = criterion_ebadv_G:forward(output_adv, g_final)
      -- backward pass
      -- feedback to loss layer, separately
    df_D = criterion_ebadv_G:backward(output_adv, g_final)    
    model_D:backward(g_final, df_D)
    df_adv = model_D.modules[1].gradInput
      
    -- (2.2) reconstruction loss (L2 loss)
      -- compare to the actual input that feed into the generator/decomposer
    f_mrecon = criterion_maskrecon:forward({g_synth,g_inputs2_mask}, g_inputs2_img)
    df_mrecon_full = criterion_maskrecon:backward({g_synth, g_inputs2_mask}, g_inputs2_img)
    df_mrecon = df_mrecon_full[1]
    f_synth = f_mrecon*4
    df_synth = df_mrecon*4

    -- (2.3) loss on A, not defined yet
    f_A_smooth = criterion_A_tv_partial:forward(g_A, g_inputs2_mask)*0.2
    df_A_smooth = criterion_A_tv_partial:backward(g_A,  g_inputs2_mask)*0.2
    f_A = f_A_smooth
    df_A = df_A_smooth
 
    -- (2.4) loss on N, L2 loss with the fitted geometry ('pseudo ground truth')     
    -- preprocess N with mask 
    f_N_sim = criterion_N:forward(g_N, g_inputs2_geo)*0.2
    df_N_sim = criterion_N:backward(g_N, g_inputs2_geo)*0.2
    f_N_smooth = criterion_N_smooth_partial:forward(g_N, g_inputs2_mask)*0.2
    df_N_smooth = criterion_N_smooth_partial:backward(g_N, g_inputs2_mask)*0.2

    f_N = f_N_sim + f_N_smooth 
    df_N = df_N_sim + df_N_smooth 

    -- Nnm loss, force the norm of normal to be 1
    local gt_Nnm = torch.Tensor(g_Nnm:size()):fill(1)
    f_Nnm = criterion_Nnm:forward(g_Nnm, gt_Nnm)*5
    df_Nnm = criterion_Nnm:backward(g_Nnm, gt_Nnm)*5

    -- (2.5) loss on L, not defined yet
    f_Lr = criterion_Lr:forward(g_Lr, g_inputs2_Lr)*5
    df_Lr = criterion_Lr:backward(g_Lr, g_inputs2_Lr)*5
    f_Lg = criterion_Lg:forward(g_Lg, g_inputs2_Lg)*5
    df_Lg = criterion_Lg:backward(g_Lg, g_inputs2_Lg)*5
    f_Lb = criterion_Lb:forward(g_Lb, g_inputs2_Lb)*5
    df_Lb = criterion_Lb:backward(g_Lb, g_inputs2_Lb)*5

    -- (2.6) loss on S, not defined yet
    f_S_bw = criterion_S_bw:forward(g_S, g_inputs2_mask)
    df_S_bw = criterion_S_bw:backward(g_S, g_inputs2_mask)
    --f_S_smooth = criterion_S_smooth:forward(g_S, g_inputs2_mask)
    --df_S_smooth = criterion_S_smooth:backward(g_S, g_inputs2_mask)
    f_S = f_S_bw 
    df_S = df_S_bw 
    -- (2.6) loss on B, not defined yet
    f_B = criterion_B:forward(g_B, g_B)
    df_B = criterion_B:backward(g_B, g_B)

    -- (2.6) loss on M
    f_M_sim = criterion_M:forward(g_M, g_inputs2_mask)*5
    df_M_sim = criterion_M:backward(g_M, g_inputs2_mask)*5
    f_M_smooth = criterion_M_smooth:forward(g_M, g_M)*2
    df_M_smooth = criterion_M_smooth:backward(g_M, g_M)*2
    f_M = f_M_sim + f_M_smooth
    df_M = df_M_sim + df_M_smooth

    -- (2.6) loss on final
    f_final_recon = criterion_final_abs:forward(g_final, g_inputs2_img)*2
    df_final_recon = criterion_final_abs:backward(g_final, g_inputs2_img)*2
    f_final = f_adv + f_final_recon
    df_final = df_adv + df_final_recon

    -- update decoder
    optim.rmsprop(fevalDec, parameters_Dec, sgdState_Dec)

    -- update encoder
    optim.rmsprop(fevalEnc, parameters_Enc, sgdState_Enc)

    -- display progress
    xlua.progress(t, dataset:size()[1])
  end -- end for loop over dataset

  -- time taken
  time = sys.clock() - time
  time = time / dataset:size()[1]
  print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

  -- print confusion matrix
  print(confusion)
  trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
  confusion:zero()

  -- save/log current net
  if epoch % opt.saveFreq == 0 then
    local filename = paths.concat(opt.save, 'formation.t7')
    os.execute('mkdir -p ' .. sys.dirname(filename))
    if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old.t7')
    end
    print('<trainer> saving network to '..filename)
    model_D:clearState()
    model_Enc:clearState()
    model_Dec:clearState()
    torch.save(filename, {D = model_D, Enc = model_Enc, Dec = model_Dec, opt = opt})
    model_D:cuda()
  	model_Enc:cuda()
  	model_Dec:cuda()
    parameters_D,gradParameters_D = model_D:getParameters()
	  parameters_Enc,gradParameters_Enc = model_Enc:getParameters()
	  parameters_Dec,gradParameters_Dec = model_Dec:getParameters()
  end

  -- next epoch
  epoch = epoch + 1
end

--
-- test function
--
function formation.test(dataset)
  model_Enc:evaluate()
  model_Dec:evaluate()
  model_D:evaluate()
  local time = sys.clock()
  local N = N or dataset:size()[1]

  print('\n<trainer> on testing Set:')
  for t = 1,N,opt.batchSize do
    -- display progress
    xlua.progress(t, dataset:size()[1])

    ----------------------------------------------------------------------
    --(1) Real data
    math.randomseed( os.time() )
    local inputs = torch.Tensor(opt.batchSize,opt.geometry[1],opt.geometry[2], opt.geometry[3])
    local targets = torch.ones(opt.batchSize)
    local k = 1
    for i = t,math.min(t+opt.batchSize-1,dataset:size()[1]) do
      local idx = math.random(dataset:size()[1])
      local sample = dataset[idx]
      local sample_img = sample:narrow(1,1,3)
      local input = sample_img:clone()
      inputs[k] = input
      k = k + 1
    end
    local preds = model_D:forward(inputs) -- get predictions from D
    energy_real = criterion_ebadv_test:forward(inputs,preds)

    ----------------------------------------------------------------------
    -- (2) Generated data (don't need this really, since no 'validation' generations)
    local inputs_fake_img = torch.Tensor(opt.batchSize, opt.geometry[1], opt.geometry[2], opt.geometry[3])
    local inputs_fake_mask = torch.Tensor(opt.batchSize, opt.geometry[1], opt.geometry[2], opt.geometry[3])
    math.randomseed( os.time() )
    for i = 1,opt.batchSize do
      local idx = math.random(dataset:size()[1])
      local sample = dataset[idx]
      local sample_img = sample:narrow(1,1,3)
      local sample_mask = sample:narrow(1,7,1)
      sample_mask = torch.repeatTensor(sample_mask,3,1,1)
      inputs_fake_img[i] = sample_img:clone()
      inputs_fake_mask[i] = sample_mask:clone()
    end 
      -- forward pass: Enc, from input image to code z
    local sample_g_zA, sample_g_zN, sample_g_zL, sample_g_zB, sample_g_zM = unpack(model_Enc:forward(inputs_fake_img))
    local inputs = unpack(model_Dec:forward({sample_g_zA, sample_g_zN, sample_g_zL, sample_g_zB, sample_g_zM, inputs_fake_mask}))
    local targets = torch.zeros(opt.batchSize)
    local preds = model_D:forward(inputs) -- get predictions from D

    energy_fake = criterion_ebadv_test:forward(inputs,preds)
  end -- end loop over dataset

  -- timing
  time = sys.clock() - time
  time = time / dataset:size()[1]
  print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')
  print("Real data energy is " .. (energy_real))
  print("Fake data energy is " .. (energy_fake))

end

return formation