import sys
from src import *

valid = ['train', 'summary', 'test']

if(sys.argv[1] not in valid):
    print("Unrecognized command. Only train, summary, and test are valid.")
    sys.exit()

args = Options().parse()
seed = 123

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

plNetwork = PatchLevelNetwork()
ilNetwork = ImageLevelNetwork()

if sys.argv[1] == valid[0]:
    if __name__ == '__main__':
        pl = PatchLevel(args, plNetwork)
        pl.train()

        il = ImageLevel(args, plNetwork, ilNetwork)
        il.train()
elif sys.argv[1] == valid[1]:
    if __name__ == '__main__':
        pl = PatchLevel(args, plNetwork)
        pl.validate()

        il = ImageLevel(args, plNetwork, ilNetwork)
        il.validate(roc=True)
else:
    if __name__ == '__main__':
        pl = PatchLevel(args, plNetwork)
        pl.test(args.test_path)

        il = ImageLevel(args, plNetwork, ilNetwork)
        il.test(args.test_path, ensemble=args.ensemble == 1)
