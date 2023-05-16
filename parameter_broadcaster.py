import os

from grid2op import make
from grid2op.Action import TopologyChangeAndDispatchAction
from grid2op.Reward import CombinedScaledReward, GameplayReward, L2RPNReward

from MyEnv import makeMyEnv, makeCustomeEnv
from d4pg_args import parser
from utils import Q, Q_compress, DQNnetwork
import numpy as np
import reverb
import sys
import time
import zlib
import pickle
import pytorch_actors


def get_shutdown_status(client, shutdown_table_name):
    sample = next(client.sample(shutdown_table_name))[0]
    return int(sample.data[0])

def get_weights_to_broadcast(client, broadcast_table_name):
    print("Waiting for weights", time.time())
    sample = next(client.sample(broadcast_table_name))[0]
    print("Received weights", time.time())
    return sample.data
DATA_DIR = "./data"
if __name__ == "__main__":
    args = parser.parse_args()

    print(vars(args))

    # Initialize environment
    env_path = os.path.join(DATA_DIR, args.taskstr)
    environment_grid, _ = makeCustomeEnv(args.taskstr, args.seed)
    model_sizes = tuple([int(x) for x in args.model_str.split(",")])
    DQN_network = DQNnetwork(environment_grid,environment_grid.action_space)
    policy_network = DQN_network.make_networks(placement=args.learner_device_placement,
                                               policy_layer_sizes=model_sizes)["policy"]
    observation_spec = DQN_network.observation_size
    action_spec = DQN_network.action_size

    # Create pytorch model (we send the state dict to the actors)
    pytorch_model = pytorch_actors.create_model(observation_spec,
                                                action_spec,
                                                args.sigma,
                                                policy_layer_sizes=model_sizes)
    
    address = "localhost:%d" % args.port
    client = reverb.Client(address)

    def quantize_and_broadcast_weights(weights, id):

        print("Broadcasting weights", time.time())
        
        # Quantize weights artificially
        tstart = time.time()
        weights = [Q(x, args.quantize_communication) for x in weights]
        #weights = [Q_opt(x, args.quantize_communication) for x in weights]
        print("Quantized (t=%f)" % (time.time()-tstart))

        # Load weights into pytorch model
        tstart = time.time()
        pytorch_actors.pytorch_model_load_state_dict(pytorch_model, weights)
        print("Load weights (t=%f)" % (time.time()-tstart))

        # Quantize
        tstart = time.time()
        quantized_actor = pytorch_actors.pytorch_quantize(pytorch_model, args.quantize)
        print("Pytorch quantized (t=%f)" % (time.time()-tstart))

        # State dict
        state_dict = quantized_actor.state_dict()
        # if args.weight_compress != 0:
        #     for k,v in state_dict.items():
        #         state_dict[k] = Q_compress(state_dict[k], args.weight_compress)
        state_dict["id"] = id

        # Send over packed params to avoid overhead
        tstart = time.time()
        weights = [pickle.dumps(state_dict)]
        if args.compress:
            weights = [zlib.compress(x) for x in weights]
        weights = [np.fromstring(x, dtype=np.uint8) for x in weights]
        print("Compress %f" % (time.time()-tstart))

        client.insert(weights,
                      {args.model_table_name : 1.0})

        print("Done broadcasting", time.time())

    c = 0
    while True:        
        should_shutdown = get_shutdown_status(client, args.shutdown_table_name)
        sys.stdout.flush()
        if should_shutdown:
            break

        weights = get_weights_to_broadcast(client, args.broadcaster_table_name)
        quantize_and_broadcast_weights(weights, c)
        
        c += 1
