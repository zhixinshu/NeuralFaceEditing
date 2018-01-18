require 'nn'

local SACompose, parent = torch.class('nn.SACompose', 'nn.Module')

-- The synthesis layer using shperical harmonics

function SACompose:__init(dimension)
    -- ??? how to initialize ???
    parent.__init(self)
    self.gradInput = {}
    self.nSample = 1
    self.nPixel = 1
end

local function sleep(n)
  os.execute("sleep " .. tonumber(n))
end

function SACompose:updateOutput(input)

    -- Inputs are A (albedo), L (lighting) and N (normals)
    -- A: 3 channels image, R, G, B, (tensor size 3xWxH)
    -- N: 3 channels image, Nx, Ny, Nz, (tensor size 3xWxH)
    -- L: 9 dimensional vector, spherical harmonics coefficient (size 9)
    local A = input[1] -- num x 3 x W x H tensor
    local S = input[2] -- num x 3 x W x H tensor
    self.output = torch.cmul(A , S)
    return self.output
end

function SACompose:updateGradInput(input, gradOutput)
    -- Verify again for correct handling of 0.5 multiplication
    self.gradInput = {}
    -- This is a layer with no parameter, gradInput = gradOutput*f'(x)

    local A = input[1]
    local S = input[2]
    
    --self.nSample= A:size(1) -- number of samples in this batch
    --self.nPixel = A:size(3)*A:size(4) -- number of pixel equals to W*H
    -- compute dIdA

    local dIdA = torch.cmul(gradOutput, S)
    self.gradInput[1] = dIdA

    -- compute dIdS 
    local dIdS = torch.cmul(gradOutput, A) -- RGB shading
    self.gradInput[2] = dIdS

    return self.gradInput
end
