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
    parser.add_argument('--quiet', type=bool, default=False, help='quiet mode')
    parser.add_argument('--skip-check', type=bool, default=False, help='skip onnx check')
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint)
    net = NNArch(checkpoint["args"])
    net.eval()
    net.load_state_dict(checkpoint["state_dict"])

    input = torch.ones((1,) + INPUT_SIZE)
    output_v, output_pi = net(input)
    if not args.quiet:
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

    if not args.skip_check:
        onnx_model = onnx.load(args.output)
        onnx.checker.check_model(onnx_model)

    if not args.quiet:
        print(f"onnxruntime device: {onnxruntime.get_device()}")
        print(f'ort avail providers: {onnxruntime.get_available_providers()}')
        ort_session = onnxruntime.InferenceSession(args.output) # , providers=["CUDAExecutionProvider"])
        print(f'{ort_session.get_providers()=}')
        print(f'{ort_session.get_inputs()=}')
        ort_output_v, ort_output_pi = ort_session.run(None, {"in": input.numpy()})
        print(f'{numpy.exp(ort_output_v)=}')