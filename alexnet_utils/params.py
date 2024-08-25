import argparse

# Custom function to convert a string to a tuple
def tuple_type(arg):
    try:
        values = arg.split(',')
        result = tuple(int(value) for value in values)
        return result

    except ValueError:
        raise argparse.ArgumentTypeError("Invalid tuple value: {}. Expected format: x,y,z".format(arg))

def print_arguments(parser, args):
    print("Arguments and Data Types:")
    for action in parser._actions:
        if isinstance(action, argparse._StoreAction):
            arg_name = action.dest
            arg_type = action.type.__name__ if action.type else "None"
            arg_value = getattr(args, arg_name)
            
            print(f"  {arg_name}: {arg_type} - {arg_value}")

parser = argparse.ArgumentParser()

parser.add_argument('-target_size', type=tuple_type, default=(240,240), help="target size to resize images to before training")
parser.add_argument('-batch_size', type=int, default=16, help="batch size for training")
parser.add_argument('-train_frac', type=float, default=0.80, help="fraction to use for the train sample")
parser.add_argument('-random_state', type=int, default=42, help="seed for random processes for reproducibility")
parser.add_argument('-num_classes', type=int, default=2, help="Number of classes")
parser.add_argument('-channels', type=int, default=3, help="Number of channels in the image data")
parser.add_argument('-output_dir', default="output", help="Location to store the outputs generated during training")
parser.add_argument('-augmentation_types', nargs='+', type=str, default=['flip', 'rotation'], choices=['brightness', 'contrast', 'rotation', 'flip', 'None'],
                    help='Types of augmentation: brightness, contrast')


if __name__=="__main__":

    args = parser.parse_args()
    print_arguments_and_types(parser, args)
