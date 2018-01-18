require 'nn'

local MarginNegMSECriterion, parent = torch.class('nn.MarginNegMSECriterion', 'nn.Criterion')

function MarginNegMSECriterion:__init(strength)
    parent.__init(self)
    self.margin = 20
end

local function sleep(n)
  os.execute("sleep " .. tonumber(n))
end

function MarginNegMSECriterion:updateOutput(input, target)

    local mse = nn.MSECriterion()
    local loss1 = mse:forward(input,target)
    local loss2 = self.margin - loss1
    if loss2 > 0 then
        self.output = loss2
    else
        self.output = 0
    end
    return self.output

end

-- TV loss backward pass inspired by kaishengtai/neuralart
function MarginNegMSECriterion:updateGradInput(input, target)
    
    local eps = 1e-8
    local grad = input:clone()
    grad:fill(0)
    local mse = nn.MSECriterion()
    local loss1 = mse:forward(input,target)
    local loss2 = self.margin - loss1
    if loss2 > 0 then
        grad = mse:backward(input,target)*(-1)
    else
        gard = grad:fill(eps)*(-1)
    end
    
    self.gradInput =  grad
    return self.gradInput
end