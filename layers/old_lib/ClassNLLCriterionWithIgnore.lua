require 'lib_ClassNLLCriterionWithIgnore'

--local THNN = require 'nn.THNN'
local ClassNLLCriterionWithIgnore, parent = torch.class('nn.ClassNLLCriterionWithIgnore', 'nn.Criterion')

function ClassNLLCriterionWithIgnore:__init(weights, sizeAverage)
    parent.__init(self)
    if sizeAverage ~= nil then
       self.sizeAverage = sizeAverage
    else
       self.sizeAverage = true
    end
    if weights then
       assert(weights:dim() == 1, "weights input should be 1-D Tensor")
       self.weights = weights
    end

    self.output_tensor = torch.zeros(1)
    self.total_weight_tensor = torch.ones(1)
    self.target = torch.zeros(1):long()
end

function ClassNLLCriterionWithIgnore:__len()
   if (self.weights) then
      return #self.weights
   else
      return 0
   end
end

function ClassNLLCriterionWithIgnore:updateOutput(input, target)
   if type(target) == 'number' then
      if input:type() ~= 'torch.CudaTensor' then
         self.target = self.target:long()
      end
      self.target[1] = target
   elseif target:type() == 'torch.CudaTensor' then
      self.target = target
   else
      self.target = target:long()
   end

   --input.THNN.ClassNLLCriterionWithIgnore_updateOutput(
   --   input:cdata(),
   --   self.target:cdata(),
   --   self.output_tensor:cdata(),
   --   self.sizeAverage,
   --   THNN.optionalTensor(self.weights),
   --   THNN.optionalTensor(self.total_weight_tensor)
   --)
   self.output = self.output_tensor[1]
   return self.output, self.total_weight_tensor[1]
end

function ClassNLLCriterionWithIgnore:updateGradInput(input, target)
   if type(target) == 'number' then
      self.target[1] = target
   elseif target:type() == 'torch.CudaTensor' then
      self.target = target
   else
      self.target = target:long()
   end

   self.gradInput:resizeAs(input):zero()

   lib_ClassNLLCriterionWithIgnore.updateGradInput(
      input:cdata(),
      self.target:cdata(),
      self.gradInput:cdata(),
      self.sizeAverage,
      self.weights:cdata(),
      self.total_weight_tensor:cdata()
   )

   return self.gradInput
end
