import math
import torch
import torch.nn as nn
from layers import CustomConv2D, CustomMaxPool2D, CustomLinear, CustomCrossEntropyLoss

def test_maxpool_custom():
    # Test Case 1: Basic 2x2 pool, stride=2 (no padding)
    x1 = torch.tensor([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ]]], dtype=torch.float32)
    
    pool1 = CustomMaxPool2D(kernel_size=2, stride=2)
    out1 = pool1(x1)
    expected1 = torch.tensor([[[[6, 8], [14, 16]]]], dtype=torch.float32)
    assert torch.allclose(out1, expected1), f"Test Case 1 Failed\nGot:{out1}\nExpected:{expected1}"

    # Test Case 2: Input not divisible by kernel size (5x5 in, 3x3 kernel)
    x2 = torch.tensor([[[
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11,12,13,14,15],
        [16,17,18,19,20],
        [21,22,23,24,25]
    ]]], dtype=torch.float32)
    
    pool2 = CustomMaxPool2D(kernel_size=3, stride=2)
    out2 = pool2(x2)
    expected2 = torch.tensor([[[[13, 15], [23, 25]]]], dtype=torch.float32)
    assert torch.allclose(out2, expected2), f"Test Case 2 Failed\nGot:{out2}\nExpected:{expected2}"

    # Test Case 3: Identity pooling (kernel_size=1)
    x3 = torch.tensor([[[
        [1, 2],
        [3, 4]
    ]]], dtype=torch.float32)
    
    pool3 = CustomMaxPool2D(kernel_size=1)
    out3 = pool3(x3)
    assert torch.allclose(out3, x3), f"Test Case 3 Failed\nGot:{out3}\nExpected:{x3}"

    # Test Case 4: With padding (3x3 input, 2x2 kernel, padding=1)
    x4 = torch.tensor([[[
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]]], dtype=torch.float32)
    
    pool4 = CustomMaxPool2D(kernel_size=2, stride=1, padding=1)
    out4 = pool4(x4)
    expected4 = torch.tensor([[[
        [1, 2, 3, 3],
        [4, 5, 6, 6],
        [7, 8, 9, 9],
        [7, 8, 9, 9]
    ]]], dtype=torch.float32)
    assert torch.allclose(out4, expected4), f"Test Case 4 Failed\nGot:{out4}\nExpected:{expected4}"

    # Test Case 5: Large stride (5x5 input, 3x3 kernel, stride=3)
    x5 = torch.tensor([[[
        [ 1, 2, 3, 4, 5],
        [ 6, 7, 8, 9,10],
        [11,12,13,14,15],
        [16,17,18,19,20],
        [21,22,23,24,25]
    ]]], dtype=torch.float32)
    pool5 = CustomMaxPool2D(kernel_size=3, stride=3, padding=1)
    out5 = pool5(x5)
    expected5 = torch.tensor([[[[7, 10], [22, 25]]]], dtype=torch.float32)
    assert torch.allclose(out5, expected5), f"Test Case 5 Failed\nGot:{out5}\nExpected:{expected5}"

    # print("All test cases passed!")



def test_custom_conv2d():
    # Test Case 1: Identity kernel (output = input)
    x1 = torch.tensor([[[
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]]], dtype=torch.float32)
    
    # Create identity kernel (center=1, padding=1 to maintain size)
    conv1 = CustomConv2D(in_channels=1, out_channels=1, kernel_size=3, padding=1)
    conv1.weight.data = torch.zeros_like(conv1.weight)
    conv1.weight.data[0, 0, 1, 1] = 1  # Center element = 1
    conv1.bias.data = torch.zeros_like(conv1.bias)  # Disable bias
    
    out1 = conv1(x1)
    assert torch.allclose(out1, x1, atol=1e-6), f"Test Case 1 Failed\nGot:{out1}\nExpected:{x1}"

    # Test Case 2: Summing kernel (kernel of ones)
    x2 = torch.ones(1, 1, 2, 2)  # [[[[1, 1], [1, 1]]]]
    conv2 = CustomConv2D(in_channels=1, out_channels=1, kernel_size=2, padding=0)
    conv2.weight.data = torch.ones_like(conv2.weight)  # All ones kernel
    conv2.bias.data = torch.zeros_like(conv2.bias)
    
    out2 = conv2(x2)
    expected2 = torch.tensor([[[[4.0]]]])  # 1 * 1 + 1 * 1 + 1 * 1 + 1 * 1 = 4
    assert torch.allclose(out2, expected2, atol=1e-6), f"Test Case 2 Failed\nGot:{out2}\nExpected:{expected2}"

    # Test Case 3: Multi-channel input
    x3 = torch.ones(1, 2, 2, 2)  # 2 input channels, each [[1,1], [1,1]]
    conv3 = CustomConv2D(in_channels=2, out_channels=1, kernel_size=2, padding=0)
    conv3.weight.data = torch.ones_like(conv3.weight)  # Each channel's kernel is [[1,1],[1,1]]
    conv3.bias.data = torch.zeros_like(conv3.bias)
    
    out3 = conv3(x3)
    expected3 = torch.tensor([[[[8.0]]]])  # (1+1+1+1)*2 (sum over 2 channels)
    assert torch.allclose(out3, expected3, atol=1e-6), f"Test Case 3 Failed\nGot:{out3}\nExpected:{expected3}"

    # Test Case 4: Compare with PyTorch's Conv2d
    torch.manual_seed(42)
    x4 = torch.randn(1, 3, 5, 5)  # Random input
    # Create identical layers
    custom_conv = CustomConv2D(in_channels=3, out_channels=2, kernel_size=3, padding=1, stride=2)
    pytorch_conv = nn.Conv2d(3, 2, kernel_size=3, padding=1, stride=2, bias=True)
    
    # Copy weights and bias
    pytorch_conv.weight.data = custom_conv.weight.data.clone()
    pytorch_conv.bias.data = custom_conv.bias.data.clone()
    
    out_custom = custom_conv(x4)
    out_pytorch = pytorch_conv(x4)
    assert torch.allclose(out_custom, out_pytorch, atol=1e-6), "Test Case 4 Failed: Outputs differ"

    # Test Case 5: Kernel larger than input (with padding)
    x5 = torch.tensor([[[
        [1, 2],
        [3, 4]
    ]]], dtype=torch.float32)
    conv5 = CustomConv2D(in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=1)
    conv5.weight.data = torch.ones_like(conv5.weight)  # 3x3 kernel of ones
    conv5.bias.data = torch.zeros_like(conv5.bias)
    
    # Padded input becomes:
    # [[0, 0, 0, 0],
    #  [0, 1, 2, 0],
    #  [0, 3, 4, 0],
    #  [0, 0, 0, 0]]
    # Each 3x3 window sum:
    out5 = conv5(x5)
    expected5 = torch.tensor([[[
        [1+3, 2+4],  # Top-left window sum: 1+3=4 (others are 0?), wait need to compute manually
        [3+0, 4+0]   # Wait, the kernel is 3x3, so all windows including padding zeros.
    ]]])  # Actual sum for each position needs precise calculation
    
    # Manually compute expected output:
    # Top-left (1 valid element + 3 zeros + 3 zeros + ...) = 1+3=4? Maybe better to compute exact:
    # For input after padding (3x3 kernel around each position in original 2x2 with padding=1):
    # Positions in original x5 (with padding=1):
    # (0,0) → padded as [0,0,0], [0,1,2], [0,3,4] → sum is 0+0+0 +0+1+2 +0+3+4 = 10
    # (0,1) → window includes [0,0,0], [1,2,0], [3,4,0] → sum 1+2+3+4 = 10
    # (1,0) → [0,1,2], [0,3,4], [0,0,0] → sum 1+3 +2+4 = 10
    # (1,1) → [1,2,0], [3,4,0], [0,0,0] → sum 1+2+3+4 = 10
    # So expected5 should be a 2x2 tensor with all 10s
    expected5 = torch.tensor([[[[10.0, 10.0], [10.0, 10.0]]]])
    assert torch.allclose(out5, expected5, atol=1e-6), f"Test Case 5 Failed\nGot:{out5}\nExpected:{expected5}"



def test_custom_linear():
    # Test Case 1: Identity transformation (weight=eye, bias=0)
    x1 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    custom_lin1 = CustomLinear(2, 2, bias=False)
    custom_lin1.weight.data = torch.eye(2)  # Set weights to identity matrix
    
    out1 = custom_lin1(x1)
    assert torch.allclose(out1, x1), "Test Case 1 Failed"

    # Test Case 2: Compare with PyTorch's Linear
    torch.manual_seed(42)
    x2 = torch.randn(3, 5)  # Batch of 3 samples, 5 features each
    # Create identical layers
    custom_lin2 = CustomLinear(5, 3, bias=True)
    pytorch_lin2 = nn.Linear(5, 3, bias=True)
    
    # Copy weights and bias
    pytorch_lin2.weight.data = custom_lin2.weight.data.clone()
    pytorch_lin2.bias.data = custom_lin2.bias.data.clone()
    
    out_custom = custom_lin2(x2)
    out_pytorch = pytorch_lin2(x2)
    assert torch.allclose(out_custom, out_pytorch, atol=1e-6), "Test Case 2 Failed"

    # Test Case 3: Zero weights and bias
    x3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    custom_lin3 = CustomLinear(2, 3, bias=True)
    custom_lin3.weight.data.zero_()
    custom_lin3.bias.data.zero_()
    
    out3 = custom_lin3(x3)
    expected3 = torch.zeros(2, 3)
    assert torch.allclose(out3, expected3), "Test Case 3 Failed"

    # Test Case 4: Multi-dimensional input
    x4 = torch.randn(2, 3, 4)  # Shape (2, 3, 4)
    custom_lin4 = CustomLinear(4, 6)
    pytorch_lin4 = nn.Linear(4, 6)
    
    # Copy parameters
    pytorch_lin4.weight.data = custom_lin4.weight.data.clone()
    pytorch_lin4.bias.data = custom_lin4.bias.data.clone()
    
    out_custom4 = custom_lin4(x4)
    out_pytorch4 = pytorch_lin4(x4)
    assert torch.allclose(out_custom4, out_pytorch4, atol=1e-6), "Test Case 4 Failed"

    # Test Case 5: No bias
    x5 = torch.randn(5, 10)
    custom_lin5 = CustomLinear(10, 5, bias=False)
    pytorch_lin5 = nn.Linear(10, 5, bias=False)
    
    pytorch_lin5.weight.data = custom_lin5.weight.data.clone()
    assert torch.allclose(custom_lin5(x5), pytorch_lin5(x5)), "Test Case 5 Failed"

    # print("All test cases passed!")


def test_custom_ce_loss():
    # Test Case 1: Compare with PyTorch's CrossEntropyLoss (mean reduction)
    torch.manual_seed(42)
    logits = torch.randn(3, 5)  # Batch size 3, 5 classes
    targets = torch.tensor([0, 2, 4])  # Target class indices

    custom_loss = CustomCrossEntropyLoss(reduction='mean')
    torch_loss = nn.CrossEntropyLoss(reduction='mean')

    out_custom = custom_loss(logits, targets)
    out_torch = torch_loss(logits, targets)
    assert torch.allclose(out_custom, out_torch, atol=1e-6), "Test Case 1 Failed"

    # Test Case 2: Multi-dimensional input (images)
    logits = torch.randn(2, 10, 5, 5)  # Batch 2, 10 classes, 5x5 spatial
    targets = torch.randint(0, 10, (2, 5, 5))

    out_custom = custom_loss(logits, targets)
    out_torch = torch_loss(
        logits.view(2, 10, -1).transpose(1, 2).contiguous().view(-1, 10),
        targets.view(-1)
    )
    assert torch.allclose(out_custom, out_torch, atol=1e-6), "Test Case 2 Failed"

    # Test Case 3: All logits equal (edge case)
    logits = torch.ones(2, 3)  # All logits = 1
    targets = torch.tensor([0, 1])
    loss = custom_loss(logits, targets)
    # Log-softmax for logits=1,1,1: log(1/3) ≈ -1.0986
    expected = -torch.tensor([-1.0986, -1.0986]).mean()
    assert torch.allclose(loss, expected, atol=1e-4), "Test Case 3 Failed"

    # print("All test cases passed!")


import argparse
# import math
# import torch
# import torch.nn as nn
# from layers import CustomConv2D, CustomMaxPool2D, CustomLinear, CustomCrossEntropyLoss

def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description='Autograder for custom neural network components',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 添加问题选择参数
    parser.add_argument('-q', '--question', 
                       type=str, 
                       choices=['q1', 'q2', 'q3', 'q4', 'all'],
                       default='all',
                       help='Select question to grade (q1-q4) or grade all')
    
    # 解析参数
    args = parser.parse_args()
    
    # 问题映射字典
    question_map = {
        'q1': test_custom_linear,
        'q2': test_custom_conv2d,
        'q3': test_maxpool_custom,
        'q4': test_custom_ce_loss,
    }
    
    # 执行测试
    try:
        if args.question == 'all':
            print("Grading all questions:")
            for name, test_func in question_map.items():
                print(f"\nRunning {name}...")
                test_func()
        else:
            print(f"Grading {args.question}:")
            question_map[args.question]()
            
        print("\nAll selected tests passed!")
        
    except AssertionError as e:
        print(f"\nTest failed: {str(e)}")
        exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        exit(2)

if __name__ == '__main__':
    main()