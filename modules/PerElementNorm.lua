require 'nn'

local PerElementNorm, parent = torch.class('nn.PerElementNorm', 'nn.Module')

-- The synthesis layer using shperical harmonics

function PerElementNorm:__init(dimension)
    -- ??? how to initialize ???
    parent.__init(self)
end

local function sleep(n)
  os.execute("sleep " .. tonumber(n))
end

function PerElementNorm:updateOutput(input)

    -- Input: N --  normal map
    -- Output: Nnn -- per element norm of the normal
    -- N: 3 channels image, Nx, Ny, Nz, (tensor size 3xWxH)
    -- Nnn: 1 channel image as per se
    local N = input -- num x 3 x W x H tensor
    local Nnn = torch.norm(N,2,2) -- L2 norm on 2nd dimension
    Nnn = torch.pow(Nnn,2) -- square of it   
    self.output = Nnn
    return self.output
end

function PerElementNorm:updateGradInput(input, gradOutput)
    local dNnn = input:clone()
    for i=1,3 do
        dNnn[{{},{i},{},{}}] = torch.cmul(dNnn[{{},{i},{},{}}],gradOutput)
    end
    self.gradInput = dNnn*2
    return self.gradInput
end

