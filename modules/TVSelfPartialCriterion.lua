require 'nn'

local TVSelfPartialCriterion, parent = torch.class('nn.TVSelfPartialCriterion', 'nn.Criterion')

function TVSelfPartialCriterion:__init(strength)
    parent.__init(self)
    self.strength = 1
    --self.x_diff = torch.Tensor()
    --self.y_diff = torch.Tensor()
    self.nSample = 1
end

local function sleep(n)
  os.execute("sleep " .. tonumber(n))
end

function TVSelfPartialCriterion:updateOutput(input,mask)

    local input_mask = torch.cmul(input,mask)

    local x_diff = input_mask[{{}, {}, {1, -2}, {1, -2}}] - input_mask[{{}, {}, {1, -2}, {2, -1}}]
    local y_diff = input_mask[{{}, {}, {1, -2}, {1, -2}}] - input_mask[{{}, {}, {2, -1}, {1, -2}}]

    local m = self.strength
    m = m/input:nElement()

    local loss = m*(x_diff:norm(1)+y_diff:norm(1))

    self.output = loss
    return self.output

end

-- TV loss backward pass inspired by kaishengtai/neuralart
function TVSelfPartialCriterion:updateGradInput(input,mask)
    
    
    
    local m = self.strength
    m = m/input:nElement()

    local x_diff = input[{{}, {}, {1, -2}, {1, -2}}] - input[{{}, {}, {1, -2}, {2, -1}}]
    local y_diff = input[{{}, {}, {1, -2}, {1, -2}}] - input[{{}, {}, {2, -1}, {1, -2}}]


    local grad = input.new():resize(input:size()):zero()
    
    grad[{{}, {}, {1, -2}, {1, -2}}]:add(torch.sign(x_diff)):add(torch.sign(y_diff))
    grad[{{}, {}, {1, -2}, {2, -1}}]:add(-1, torch.sign(x_diff))
    grad[{{}, {}, {2, -1} ,{1, -2}}]:add(-1, torch.sign(y_diff))

    grad = torch.cmul(grad,mask)

    self.gradInput = grad*m
    
    return self.gradInput
end