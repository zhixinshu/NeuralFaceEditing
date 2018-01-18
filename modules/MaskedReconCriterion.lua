require 'nn'

local MaskedReconCriterion, parent = torch.class('nn.MaskedReconCriterion', 'nn.Criterion')

function MaskedReconCriterion:__init(strength)
    parent.__init(self)
    self.strength = 1
    --self.x_diff = torch.Tensor()
    --self.y_diff = torch.Tensor()
    self.nSample = 1
end

local function sleep(n)
  os.execute("sleep " .. tonumber(n))
end

function MaskedReconCriterion:updateOutput(input,target)
    --input[1] input
    --input[2] mask
    local input_mask = torch.cmul(input[1],input[2])
    local target_mask = torch.cmul(target,input[2])
    local mse = nn.MSECriterion()

    self.output = mse:forward(input_mask, target_mask)
    return self.output

end

-- TV loss backward pass inspired by kaishengtai/neuralart
function MaskedReconCriterion:updateGradInput(input,target,mask)
    
    local grad = {}
    local input_mask = torch.cmul(input[1],input[2])
    local target_mask = torch.cmul(target,input[2])
    local mse = nn.MSECriterion()

    grad[1] = mse:backward(input_mask, target_mask)
    grad[2] = torch.Tensor(input[2]:size()):fill(0)

    self.gradInput = grad
    return self.gradInput
end