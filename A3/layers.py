import torch
import torch.nn as nn
import math

"""
Allowed functions/operation:

Basic arithmetic operations: 
    - Creation Ops: https://pytorch.org/docs/stable/torch.html#indexing-slicing-joining-mutating-ops
    - Indexing, Slicing, Joining, Mutating Ops: https://pytorch.org/docs/stable/torch.html#indexing-slicing-joining-mutating-ops

Some advanced functions:
    - nn.functional.unfold: https://pytorch.org/docs/stable/generated/torch.nn.functional.unfold.html
    - torch.einsum: https://pytorch.org/docs/stable/generated/torch.einsum.html
"""

########################################## DECLARE #####################################################
# You must declare with ONE of the following statements if you have used any GenAI tools:              #
#   - I did not use any AI technologies in preparing or writing up this assignment.                    #
#   - I acknowledge the use of <insert AI system(s) and link> to generate initial ideas for            #
#       background research in the drafting of this assignment.                                        #
#   - I acknowledge the use of <insert AI system(s) and link> to generate materials that were          #
#       included within my final assignment in its modified form.                                      #
# e.g.                                                                                                 #
#   I acknowledge the use of ChatGPT <https://chatgpt.hku.hk/> to generate initial math formula        #
#   for convolution. I then use the                                                                    #
#                                                                                                      #
# If you have used GenAI tool(s), you must (i) name the tool(s), (ii) describe how it/they were used,  #
# AND (iii) reference the tool(s):                                                                     #
#                                                                                                      #
########################################################################################################
# I acknowledge the use of Deepseek and ChatGPT to generate materials that were included within my final assignment in its modified form.


class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        # Initialization code remains unchanged
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialization remains unchanged
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """
        Implement linear transformation y = xW^T + b
        Allowed functions/operations:
        - torch.einsum()
        - Tensor.view()
        - Basic arithmetic operations
        """
        ###########################################################################
        # TODO: Process input to produce output with shape:                       #
        # (..., out_features) where ... preserves input dimensions                #
        #                                                                         #
        # HINT: Consider efficient tensor operations for matrix multiplication    #
        # Student's implementation here                                           #
        ###########################################################################

        input = x
        output = torch.einsum('...i, ji->...j', input, self.weight) # for matrix multiplication
        
        # check if bias is used
        if self.bias is not None:
            output += self.bias
        return output
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################


class CustomConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        # Initialization code remains unchanged
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        # Initialization remains unchanged
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """
        Implement 2D convolution using tensor operations
        Allowed functions/operations:
        - torch.Tensor.shape
        - Tensor.view() 
        - torch.einsum()
        - nn.functional.unfold()
        - Basic arithmetic operations
        """
        ################################################################################
        # TODO: Transform input using allowed operations to produce output with shape: #
        # (N, out_channels, H_out, W_out)                                              #
        #                                                                              #
        # HINT: Consider how to reshape the weight matrix and process unfolded patches #
        #                                                                              #
        # Student's implementation here                                                #
        ################################################################################

        # take input image and break into small patches(kernel size)
        unfolded = torch.nn.functional.unfold(x, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)

        new_weight = self.weight.view(self.out_channels, -1)# reshape the weight from 4d to 2d

        output = torch.einsum('ij, bjp->bip', new_weight, unfolded)

        # check if bias is used
        if self.bias is not None:
            output += self.bias.view(1, -1, 1)

        n, h, w = x.size(0), x.size(2), x.size(3)
        final_h = (h + 2*self.padding - self.kernel_size) // self.stride + 1
        final_w = (w + 2*self.padding - self.kernel_size) // self.stride + 1
        final_output = output.view(n, self.out_channels, final_h, final_w) # reshape the output

        return final_output
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################


class CustomMaxPool2D(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
    
    def forward(self, x):
        """
        Implement 2D max pooling using tensor operations
        Allowed functions/operations:
        - torch.Tensor.shape
        - Tensor.view() 
        - Tensor.max()
        - nn.functional.unfold()
        """
        ###########################################################################
        # TODO: Process input to produce output with shape:                       #
        # (N, C, H_out, W_out)                                                    #
        #                                                                         #
        # HINT: Consider how to extract and process local windows                 #
        #                                                                         #
        # Student's implementation here                                           #
        ############################################################################
        unfolded = torch.nn.functional.unfold(x, kernel_size=self.kernel_size,stride=self.stride, padding=self.padding)
        n, c, l = unfolded.shape
        unfolded = unfolded.view(n, -1, self.kernel_size*self.kernel_size, l)

        max_pooled, _ = unfolded.max(dim=2)

        h, w = x.size(2), x.size(3)
        final_h = (h + 2*self.padding - self.kernel_size) // self.stride + 1
        final_W = (w + 2*self.padding - self.kernel_size) // self.stride + 1

        return max_pooled.view(n, -1, final_h, final_W)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################


class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Implement cross entropy loss with numerical stability
        Allowed functions/operations:
        - Tensor.max(), .exp(), .log(), .sum()
        - torch.gather()
        - Basic arithmetic operations
        - Reduction methods (mean(), sum())
        """
        ###########################################################################
        # TODO: Compute loss without using nn.CrossEntropyLoss                    #
        #                                                                         #
        # HINT: Consider numerical stability when working with exponents          #
        #                                                                         #
        # Student's implementation here                                           #
        ###########################################################################

        if logits.dim() > 2: # reshape logits to 2D
            # reshape logits and targets
            logits = logits.view(logits.size(0), logits.size(1), -1)
            logits = logits.transpose(1, 2)
            logits = logits.contiguous()
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)

        logits_max = logits.max(dim=-1, keepdim=True).values
        shifted_logits = logits - logits_max # subtract max value from each row of logits to avoid overflow when take exponent

        exp_logits = shifted_logits.exp() # take the exponent
        sum_exp = exp_logits.sum(dim=-1, keepdim=True) # get the sum of exponentials
        log_softmax = shifted_logits - sum_exp.log() # get the log softmax

        loss = -log_softmax.gather(1, targets.unsqueeze(1)).squeeze(1)

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
