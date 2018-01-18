require 'nn'

local BatchWhiteShadingCriterion, parent = torch.class('nn.BatchWhiteShadingCriterion', 'nn.Criterion')

function BatchWhiteShadingCriterion:__init(strength)
    --self.strength = 1
    self.anchor = 0.75
    self.eps = 1e-6
    parent.__init(self)
end

local function sleep(n)
  os.execute("sleep " .. tonumber(n))
end

function BatchWhiteShadingCriterion:updateOutput(input,mask)

    local input_mask = torch.cmul(input,mask)
    mask_single_channel = mask[{{},{1},{},{}}]
    local m = torch.sum(mask_single_channel) + self.eps

    local shade_r = input_mask[{{},{1},{},{}}]
    local shade_g = input_mask[{{},{2},{},{}}]
    local shade_b = input_mask[{{},{3},{},{}}]

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
function BatchWhiteShadingCriterion:updateGradInput(input, mask)
    
    local input_mask = torch.cmul(input,mask)
    mask_single_channel = mask[{{},{1},{},{}}]
    local m = torch.sum(mask_single_channel) + self.eps
 
    local shade_r = input_mask[{{},{1},{},{}}]
    local shade_g = input_mask[{{},{2},{},{}}]
    local shade_b = input_mask[{{},{3},{},{}}]

    local avg_r = torch.sum(shade_r)/m
    local avg_g = torch.sum(shade_g)/m
    local avg_b = torch.sum(shade_b)/m


    local grad = input.new():resize(input:size()):zero()
    
    grad[{{},{1},{},{}}]:fill((avg_r - self.anchor)/m)
    grad[{{},{2},{},{}}]:fill((avg_g - self.anchor)/m)
    grad[{{},{3},{},{}}]:fill((avg_b - self.anchor)/m)

    self.gradInput = torch.cmul(grad,mask)
    
    return self.gradInput
end