import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess

# file_100 = np.load("/Users/maziargomrokchi/test_data/seed_100/BCQ_Hopper-v2_20_100_1000000.npy")
# file_5 = np.load  ("/Users/maziargomrokchi/test_data/seed_5/BCQ_Hopper-v2_20_5_1000000.npy")
# file_75 = np.load("/Users/maziargomrokchi/test_data/seed_75/BCQ_Hopper-v2_200_75_1000000.npy")
# file_700 = np.load("/Users/maziargomrokchi/test_data/seed_700/BCQ_Hopper-v2_200_700_1000000.npy")
# plt.plot(file_100, label="seed_100")
# plt.plot(file_5, label="seed_5")
# plt.plot(file_75, label="seed_75")
# plt.plot(file_700, label="seed_700")
# plt.legend(loc="upper left")
# plt.show()

env = "Hopper-v2"
trj_len = 500
gen_buffer_size = 10000000
# env_seed = [20, 20, 20, 20, 20, 20, 20, 20]  # Has to be the same size as seed
# env_seed = [200, 200, 200, 200, 200, 200, 200, 200]
env_seed = [400, 400, 400, 400, 400, 400, 400, 400]
seed = [100, 5, 75, 700, 80, 500, 45, 90]

path = '/Users/maziargomrokchi/test_data'

for i in range(0, len(seed)):
    file_path = path + f"/{env}/{trj_len}/{gen_buffer_size}/{env_seed[i]}/{seed[i]}"
    if not os.path.exists(file_path):
        raise Exception(f"{file_path} does not exist!")
    os.chdir(file_path)
    performance = np.load(f"BCQ_Hopper-v2_{env_seed[i]}_{seed[i]}_1000000.npy")
    # performance = np.load(
    #     f"samin@beluga.calculquebec.ca:/home/samin/projects/rrg-dprecup/samin/learning_output/Hopper-v2/1000000/"
    #     f"{gen_buffer_size}/{env_seed[i]}/{seed[i]}/{trj_len}/results/BCQ_Hopper-v2_"
    #     f"{env_seed[i]}_{seed[i]}_1000000.npy")
    # url = subprocess.Popen(['scp', 'samin', 'Mywoodencottage1',
    #                         f"samin@beluga.calculquebec.ca:/home/samin/projects/rrg-dprecup/samin/learning_output/Hopper-v2/1000000/"
    #                         f"{gen_buffer_size}/{env_seed[i]}/{seed[i]}/{trj_len}/results/BCQ_Hopper-v2_"
    #                         f"{env_seed[i]}_{seed[i]}_1000000.npy"], stdout=subprocess.PIPE)
    # url = \
    #     f"samin@beluga.calculquebec.ca:/home/samin/projects/rrg-dprecup/samin/learning_output/Hopper-v2/1000000/" \
    #     f"{gen_buffer_size}/{env_seed[i]}/{seed[i]}/{trj_len}/results/BCQ_Hopper-v2_{env_seed[i]}_{seed[i]}_1000000.npy"
    # performance = requests.get(url, auth=HTTPBasicAuth('samin', 'Mywoodencottage1'))
    plt.plot(performance, label=f"env_seed{env_seed[i]}_seed{seed[i]}")

plt.legend(loc="upper left")
plt.show()


# vars()[f"url_envseed{env_seed[i]}_seed{seed[i]}"]