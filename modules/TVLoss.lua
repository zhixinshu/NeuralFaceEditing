require 'nn'

local TVLoss, parent = torch.class('nn.TVLoss', 'nn.Module')

function TVLoss:__init(strength)
    parent.__init(self)
    self.strength = 0
    --self.x_diff = torch.Tensor()
    --self.y_diff = torch.Tensor()
    self.nSample = 1
end

function TVLoss:updateOutput(input)
    self.output = input
    return self.output
end

-- TV loss backward pass inspired by kaishengtai/neuralart
function TVLoss:updateGradInput(input, gradOutput)
    --self.gradInput = {}
    local tvg = input:clone()
    --tvg:fill(0)
    self.nSample = input:size(1)
    
    local C = input:size(2)
    local H = input:size(3)
    local W = input:size(4)

    --local x_diff = torch.Tesnor(C, H - 1, W - 1):fill(0)
    --local y_diff = torch.Tensor(C, H - 1, W - 1):fill(0)

    for j = 1, self.nSample do

        local input_J = input[j]

        local tvg_J = input_J:clone():fill(0)

        local x_diff = torch.Tensor(3, H-1, W-1)
        local y_diff = torch.Tensor(3, H-1, W-1)


        x_diff = input_J[{{}, {1, -2}, {1, -2}}]:clone()
        x_diff = x_diff - input_J[{{}, {1, -2}, {2, -1}}]

        y_diff = input_J[{{}, {1, -2}, {1, -2}}]:clone()
        y_diff = y_diff - input_J[{{}, {2, -1}, {1, -2}}]
        
        local xy_diff = x_diff + y_diff
        tvg_J[{{}, {1, -2}, {1, -2}}] = tvg_J[{{}, {1, -2}, {1, -2}}] + xy_diff
        tvg_J[{{}, {1, -2}, {2, -1}}] = tvg_J[{{}, {1, -2}, {2, -1}}] - x_diff
        tvg_J[{{}, {2, -1}, {1, -2}}] = tvg_J[{{}, {2, -1}, {1, -2}}] - y_diff

        tvg[j] = tvg_J
    end
    tvg = tvg*self.strength
    tvg = tvg + gradOutput

    self.gradInput = tvg
    return self.gradInput
end