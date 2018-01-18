require 'nn'

local FBMCompose, parent = torch.class('nn.FBMCompose', 'nn.Module')

-- The synthesis layer using shperical harmonics

function FBMCompose:__init(dimension)
    -- ??? how to initialize ???
    parent.__init(self)
    self.gradInput = {}
    self.nSample = 1
    self.nPixel = 1
end

local function sleep(n)
  os.execute("sleep " .. tonumber(n))
end

function FBMCompose:updateOutput(input)

    -- Inputs are A (albedo), L (lighting) and N (normals)
    -- A: 3 channels image, R, G, B, (tensor size 3xWxH)
    -- N: 3 channels image, Nx, Ny, Nz, (tensor size 3xWxH)
    -- L: 9 dimensional vector, spherical harmonics coefficient (size 9)
    local F = input[1] -- num x 3 x W x H tensor
    local B = input[2] -- num x 3 x W x H tensor
    local M = input[3] -- num x 3 x W x H tensor
    local negM = M*(-1) + 1
    self.output = torch.cmul(F , M) + torch.cmul(B,negM)
    return self.output
end

function FBMCompose:updateGradInput(input, gradOutput)
    -- Verify again for correct handling of 0.5 multiplication
    self.gradInput = {}
    -- This is a layer with no parameter, gradInput = gradOutput*f'(x)

    local F = input[1] -- num x 3 x W x H tensor
    local B = input[2] -- num x 3 x W x H tensor
    local M = input[3] -- num x 3 x W x H tensor
    local negM = M*(-1) + 1
    --self.nSample= A:size(1) -- number of samples in this batch
    --self.nPixel = A:size(3)*A:size(4) -- number of pixel equals to W*H
    -- compute dIdA

    local dIdF = torch.cmul(M,gradOutput)
    self.gradInput[1] = dIdF
    local dIdB = torch.cmul(negM,gradOutput)
    self.gradInput[2] = dIdB
    local dIdM = torch.cmul((F-B),gradOutput)
    self.gradInput[3] = dIdM

    return self.gradInput
end
