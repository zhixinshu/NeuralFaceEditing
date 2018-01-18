require 'nn'

local BatchWhiteShadingCriterion2, parent = torch.class('nn.BatchWhiteShadingCriterion2', 'nn.Criterion')

function BatchWhiteShadingCriterion2:__init(strength)
    --self.strength = 1
    self.anchor = 0.75
    self.eps = 1e-6
    parent.__init(self)
end

local function sleep(n)
  os.execute("sleep " .. tonumber(n))
end

function BatchWhiteShadingCriterion2:updateOutput(input)

    local m = input:size(1)*input:size(3)*input:size(4)

    local shade_r = input[{{},{1},{},{}}]
    local shade_g = input[{{},{2},{},{}}]
    local shade_b = input[{{},{3},{},{}}]

    local avg_r = torch.sum(shade_r)/m
    local avg_g = torch.sum(shade_g)/m
    local avg_b = torch.sum(shade_b)/m

    
    local loss_r = (avg_r - self.anchor)*(avg_r - self.anchor)*0.5
    local loss_g = (avg_g - self.anchor)*(avg_g - self.anchor)*0.5
    local loss_b = (avg_b - self.anchor)*(avg_b - self.anchor)*0.5 

    local loss = loss_r + loss_g + loss_b

    self.output = loss
    return self.output

end

-- TV loss backward pass inspired by kaishengtai/neuralart
function BatchWhiteShadingCriterion2:updateGradInput(input)
    
    local m = input:size(1)*input:size(3)*input:size(4)
 
    local shade_r = input[{{},{1},{},{}}]
    local shade_g = input[{{},{2},{},{}}]
    local shade_b = input[{{},{3},{},{}}]

    local avg_r = torch.sum(shade_r)/m
    local avg_g = torch.sum(shade_g)/m
    local avg_b = torch.sum(shade_b)/m


    local grad = torch.Tensor(input:size())
    
    grad[{{},{1},{},{}}]:fill((avg_r - self.anchor)/m)
    grad[{{},{2},{},{}}]:fill((avg_g - self.anchor)/m)
    grad[{{},{3},{},{}}]:fill((avg_b - self.anchor)/m)

    self.gradInput = grad
    
    return self.gradInput
end