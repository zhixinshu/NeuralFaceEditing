require 'nn'

local TVCriterion, parent = torch.class('nn.TVCriterion', 'nn.Criterion')

function TVCriterion:__init(strength)
    parent.__init(self)
    self.strength = 0.001
    --self.x_diff = torch.Tensor()
    --self.y_diff = torch.Tensor()
    self.nSample = 1
end

local function sleep(n)
  os.execute("sleep " .. tonumber(n))
end

function TVCriterion:updateOutput(input,target)


    local gen = input:clone()
    local x_diff = gen[{{}, {}, {1, -2}, {1, -2}}] - gen[{{}, {}, {1, -2}, {2, -1}}]
    local y_diff = gen[{{}, {}, {1, -2}, {1, -2}}] - gen[{{}, {}, {2, -1}, {1, -2}}]

    local a = torch.abs(x_diff)
    local b = torch.abs(y_diff)
    --return K.sum(K.pow(a + b, 1.25))

    self.output = torch.sum(a) + torch.sum(b)

    self.output = 0*self.strength* self.output
    return self.output

end

-- TV loss backward pass inspired by kaishengtai/neuralart
function TVCriterion:updateGradInput(input, target)
    
    local gen = input:clone()
    local x_diff = torch.sign(gen[{{}, {}, {1, -2}, {1, -2}}] - gen[{{}, {}, {1, -2}, {2, -1}}])
    local y_diff = torch.sign(gen[{{}, {}, {1, -2}, {1, -2}}] - gen[{{}, {}, {2, -1}, {1, -2}}])
    local grad = gen.new():resize(gen:size()):zero()
    grad[{{}, {}, {1, -2}, {1, -2}}]:add(x_diff):add(y_diff)
    grad[{{}, {}, {1, -2}, {2, -1}}]:add(-1, x_diff)
    grad[{{}, {}, {2, -1} ,{1, -2}}]:add(-1, y_diff)
    
    self.gradInput = self.strength * grad
    return self.gradInput
end