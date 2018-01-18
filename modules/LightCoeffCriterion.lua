require 'nn'

local LightCoeffCriterion, parent = torch.class('nn.LightCoeffCriterion', 'nn.Criterion')

function LightCoeffCriterion:__init(strength)
    parent.__init(self)
end

local function sleep(n)
  os.execute("sleep " .. tonumber(n))
end

function LightCoeffCriterion:updateOutput(input,target)
    -- input: num x 10
    local input_Lcoeff = input[{{},{1,9}}]
    local mse = nn.MSECriterion()
    self.output = mse:forward(input_Lcoeff ,target)
    return self.output

end

-- TV loss backward pass inspired by kaishengtai/neuralart
function LightCoeffCriterion:updateGradInput(input,target)
    
    local grad = torch.Tensor(input:size()):fill(0)
    local input_Lcoeff = input[{{},{1,9}}]
    local mse = nn.MSECriterion()
    local loss = mse:forward(input_Lcoeff ,target)
    grad[{{},{1,9}}] = mse:backward(input_Lcoeff,target)
    self.gradInput = grad
    return self.gradInput
end