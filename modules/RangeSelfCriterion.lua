require 'nn'

local RangeSelfCriterion, parent = torch.class('nn.RangeSelfCriterion', 'nn.Criterion')

function RangeSelfCriterion:__init(strength)
    parent.__init(self)
    self.strength = 1
    self.anchor = 0.75
    self.eps = 1e-6
end

local function sleep(n)
  os.execute("sleep " .. tonumber(n))
end

function RangeSelfCriterion:updateOutput(input,mask)

    local input_mask = torch.cmul(input,mask)
    local m = torch.sum(mask) + self.eps
    local avgval = torch.sum(input_mask)/m

    local loss = (avgval - self.anchor)*(avgval - self.anchor)*0.5

    self.output = loss
    return self.output

end

-- TV loss backward pass inspired by kaishengtai/neuralart
function RangeSelfCriterion:updateGradInput(input, mask)
    
    local input_mask = torch.cmul(input,mask)
    
    local m = torch.sum(mask) + self.eps

    local avgval = torch.sum(input_mask)/m

    local grad = input.new():resize(input:size()):zero()
    
    grad:fill((avgval-self.anchor)/m)

    self.gradInput = torch.cmul(grad,mask)
    
    return self.gradInput
end