require 'nn'

local PartialSimNCriterion, parent = torch.class('nn.PartialSimNCriterion', 'nn.Criterion')

function PartialSimNCriterion:__init(strength)
    parent.__init(self)
end

local function sleep(n)
  os.execute("sleep " .. tonumber(n))
end

function PartialSimNCriterion:updateOutput(input,target,mask)
    -- input: num x 10
    local input_mask = torch.cmul(input,mask)
    local target_mask = torch.cmul(target,mask)
    local mse = nn.MSECriterion()
    self.output = mse:forward(input_mask ,target_mask)
    return self.output

end

-- TV loss backward pass inspired by kaishengtai/neuralart
function PartialSimNCriterion:updateGradInput(input,target,mask)
    
    local grad = torch.Tensor(input:size()):fill(0)
    local input_mask = torch.cmul(input,mask)
    local target_mask = torch.cmul(target,mask)
    local mse = nn.MSECriterion()
    grad = mse:backward(input_mask ,target_mask)
    self.gradInput = grad
    return self.gradInput
end