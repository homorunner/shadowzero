import torch
import torch.onnx
import numpy
import onnx
import onnxruntime
import argparse

from nnarch import NNArch, NNArgs, INPUT_SIZE

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export ONNX model')
    parser.add_argument('--checkpoint', type=str, default="checkpoint.pt", help='checkpoint file')
    parser.add_argument('--output', type=str, default="output.onnx", help='output file')
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint)
    net = NNArch(checkpoint["args"])
    net.eval()
    net.load_state_dict(checkpoint["state_dict"])

    input = torch.zeros((1,) + INPUT_SIZE)
    input[0, 0, 0, 0] = 1
    input[0, 1, 0, 1] = 1
    input[0, 2, 0, 2] = 1
    input[0, 3, 0, 3] = 1
    input[0, 4, 0, 0] = 1
    input[0, 5, 0, 1] = 1
    input[0, 6, 0, 2] = 1
    input[0, 7, 0, 3] = 1
    input[0, 8, 0, 0] = 1
    input[0, 9, 0, 1] = 1
    input[0, 10, 0, 2] = 1
    input[0, 11, 0, 3] = 1
    input[0, 12, 0, 0] = 1
    input[0, 13, 0, 1] = 1
    input[0, 14, 0, 2] = 1
    input[0, 15, 0, 3] = 1
    input[0, 16, 0, 0:4] = 1
    input[0, 17, 0, 0:4] = 1
    input[0, 18, 0, 0:4] = 1
    input[0, 19, 0, 0:4] = 1
    input[0, 20, 0, 0:4] = 1
    input[0, 21, 0, 0:4] = 1
    input[0, 22, 0, 0:4] = 1
    input[0, 23, 0, 0:4] = 1
    output_v, output_pi = net(input)
    print(torch.exp(output_v.detach()))
    print(output_pi.detach())

    inet = torch.jit.load('testdata/example_traced.pt')
    inet.eval()
    output_v, output_pi = inet(input.cuda())
    print(torch.exp(output_v.detach()))
    print(output_pi.detach())

    net.train()
    jnet = torch.jit.trace(
        net, torch.ones((1,) + INPUT_SIZE)
    )
    jnet.eval()
    output_v, output_pi = jnet(input)
    print(torch.exp(output_v.detach()))
    print(output_pi.detach())

    torch.onnx.export(net,
                      input,
                      args.output,               # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['in'],      # the model's input names
                      output_names = ['v', 'p'],    # the model's output names
                      dynamic_axes={'in' : {0 : 'batch_size'},    # variable length axes
                                    'v' : {0 : 'batch_size'},
                                    'p' : {0 : 'batch_size'}})

    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)

    print(f"onnxruntime device: {onnxruntime.get_device()}")
    print(f'ort avail providers: {onnxruntime.get_available_providers()}')
    ort_session = onnxruntime.InferenceSession(args.output) # , providers=["CUDAExecutionProvider"])
    print(f'{ort_session.get_providers()=}')
    print(f'{ort_session.get_inputs()=}')
    ort_output_v, ort_output_pi = ort_session.run(None, {"in": input.numpy()})
    print(f'{numpy.exp(ort_output_v)=}')