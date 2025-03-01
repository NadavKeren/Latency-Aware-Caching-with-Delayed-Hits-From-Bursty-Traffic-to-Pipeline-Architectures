import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random
import time

from rich import print, pretty

from types import MappingProxyType

pretty.install()

OUTPUT_FORMAT = 'pdf'

TOTAL_TICKS = 75000
NUM_OF_FREQ = 100
TICK_TIME = 1000

RECENCY_CONF = MappingProxyType({
    "MIN_OCCUR_LEN": 24,
    "MAX_OCCUR_LEN": 33,
    "MIN_TIME_BETWEEN_REQ": 700,
    "MAX_TIME_BETWEEN_REQ": 900
})

FREQ_CONF = MappingProxyType({
    "MIN_TIME_BETWEEN_REQ": TICK_TIME + 1,
    "MAX_TIME_BETWEEN_REQ": int(TICK_TIME * 1.1) + 1,
    "MIN_FIRST_OCCUR_TIME": TICK_TIME,
    "MAX_FIRST_OCCUR_TIME": TICK_TIME * 100
})

BURST_CONF = MappingProxyType({
    "MIN_LEN": 900,
    "MAX_LEN": 1100,
    "MIN_BURSTS": 70,
    "MAX_BURSTS": 80,
    "MIN_TIME_BETWEEN_BURSTS": TICK_TIME * (TOTAL_TICKS // 60),
    "MAX_FIRST_OCCUR_TIME": TICK_TIME * (TOTAL_TICKS // 100),
    "MIN_LAST_OCCUR_TIME": TICK_TIME * (TOTAL_TICKS // 100 * 95),
    "MAX_LAST_OCCUR_TIME": TICK_TIME * TOTAL_TICKS - 1500
})

TRACE_CONF = MappingProxyType({
    'RECENCY': 250000,
    'FREQUENCY': 100,
    'BURSTINESS': 100,
    'ONE_HIT_WONDERS': 150000
})

BIN_SIZE = 10
MAX_TICK_TO_SHOW = 900
TOTAL_BINS_TO_SHOW = MAX_TICK_TO_SHOW // BIN_SIZE


def create_recency_item(rng: np.random.Generator, item_num: int, latency: int, start: int = None) -> list:
    arr = []
    if start is None:
        curr_time = random.randint(0, TOTAL_TICKS * TICK_TIME 
                                   - RECENCY_CONF['MAX_OCCUR_LEN'] * (RECENCY_CONF['MAX_TIME_BETWEEN_REQ'] 
                                          + RECENCY_CONF['MIN_TIME_BETWEEN_REQ']) // 2)
    else:
        curr_time = start
    
    num_of_occurences = random.randint(RECENCY_CONF['MIN_OCCUR_LEN'], RECENCY_CONF['MAX_OCCUR_LEN'])
    inter_times = rng.integers(low = RECENCY_CONF['MIN_TIME_BETWEEN_REQ'], 
                               high = RECENCY_CONF['MAX_TIME_BETWEEN_REQ'], 
                               size=num_of_occurences)
    
    for time in inter_times:
        arr.append((curr_time, item_num, 0, latency))
        curr_time += time
    
    # for time in range(num_of_occurences):
    #     arr.append((curr_time, item_num, 0, latency))
    #     curr_time += RECENCY_CONF['MIN_TIME_BETWEEN_REQ']
        
    arr.append((curr_time, item_num, 0, latency))
    
    return arr

def create_frequency_item(rng: np.random.Generator, item_num: int, latency: int) -> list:
    arr = []
    curr_time = random.randint(FREQ_CONF['MIN_FIRST_OCCUR_TIME'], FREQ_CONF['MAX_FIRST_OCCUR_TIME'])
    
    while curr_time < TICK_TIME * TOTAL_TICKS:
        arr.append((curr_time, item_num, 0, latency))
        curr_time += random.randint(FREQ_CONF['MIN_TIME_BETWEEN_REQ'], FREQ_CONF['MAX_TIME_BETWEEN_REQ'])
        # curr_time += FREQ_CONF['MIN_TIME_BETWEEN_REQ']

    return arr

def create_burstiness_item(rng: np.random.Generator, item_num: int, latency: int) -> list:
    arr = []
    
    num_bursts = random.randint(BURST_CONF['MIN_BURSTS'], BURST_CONF['MAX_BURSTS'])
    first_burst_time = random.randint(0, BURST_CONF['MAX_FIRST_OCCUR_TIME'])
    last_burst_time = random.randint(BURST_CONF['MIN_LAST_OCCUR_TIME'], BURST_CONF['MAX_LAST_OCCUR_TIME'])
    
    time_between_bursts = random.randint(5, 10) * (last_burst_time - first_burst_time - num_bursts * TICK_TIME) // (num_bursts * 10)
    
    curr_time = first_burst_time
    for burst in range(num_bursts):
        num_of_reqs_in_burst = random.randint(BURST_CONF['MIN_LEN'], BURST_CONF['MAX_LEN'])
        # curr_time = curr_time + (last_burst_time - first_burst_time) // (num_bursts * 10) * random.randint(8, 11)
        # inter_times = rng.integers(low = BURST_CONF['MIN_TIME_BETWEEN_REQ'], high = BURST_CONF['MAX_TIME_BETWEEN_REQ'], size = num_of_reqs_in_burst)
        for ms in range(num_of_reqs_in_burst):
            arr.append((curr_time, item_num, 0, latency))
            curr_time += 1
        
        curr_time += time_between_bursts
            
        # for diff in inter_times:
        #     arr.append((curr_time, item_num, 0, latency))
        #     curr_time += diff
            
    return arr
        
    
def create_histogram(item_arr : list) -> list:  
    item_bins = np.zeros(TOTAL_BINS_TO_SHOW)
    
    for entry in item_arr:
        tick = entry[0] // TICK_TIME
        if tick < MAX_TICK_TO_SHOW:
            idx = tick // (BIN_SIZE)
            item_bins[idx] += 1
        else:
            break
        
    return item_bins


def plot_example(rng: np.random.Generator) -> None:
    plt.rcParams["figure.figsize"] = (9.5,6.3)
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['hatch.linewidth'] = 2.5
    plt.rcParams['lines.markersize'] = 10
    plt.rc('font', size=20)          # controls default text sizes
    plt.rc('axes', titlesize=20)     # fontsize of the axes title
    plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
    plt.rc('legend', fontsize=18)    # legend fontsize
    plt.rc('figure', titlesize=20)  # fontsize of the figure title
    plt.rc('axes', prop_cycle=plt.cycler(color=['#000000', '#52796F', '#84A98C']))

    recency_item = create_recency_item(rng, 1, 1000, start=550 * TICK_TIME)
    freq_item = create_frequency_item(rng, 2, 1000)
    bursty_item = create_burstiness_item(rng, 3, 1000)
    
    print(recency_item[-1][0] // TICK_TIME)
    print(freq_item[-1][0] // TICK_TIME)
    print(bursty_item[-1][0] // TICK_TIME)
    
    recency_histo = create_histogram(recency_item)
    freq_histo = create_histogram(freq_item)
    bursty_histo = create_histogram(bursty_item)
    
    times = [i * TICK_TIME * BIN_SIZE for i in range(TOTAL_BINS_TO_SHOW)]
    
    fig, ax = plt.subplots()
    
    ax.plot(times, recency_histo, label="Recent Item", linewidth=1)
    ax.plot(times, freq_histo, label="Frequent Item", linewidth=4)
    ax.plot(times, bursty_histo, label="Bursty Item", linewidth=7)
    
    ax.legend(loc="upper right", bbox_to_anchor=(1.12, 1.1),  fancybox=True, shadow=True) #  ncol=1, bbox_to_anchor=(0.5, 1.1) 
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel(f"Request Density (Requests / {BIN_SIZE} seconds)")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos : f'{x / 1000:.0f}'))
    ax.set_xlim((0, MAX_TICK_TO_SHOW*TICK_TIME))
    ax.set_yscale('log', base=10)
    fig.savefig(f"synthetic_traffic_example.{OUTPUT_FORMAT}",dpi=200)
    

def gen_trace(rng : np.random.Generator) -> None:
    print(TRACE_CONF)
    print(RECENCY_CONF)
    print(FREQ_CONF)
    print(BURST_CONF)
    
    arr = []
    count = 0
    LATENCY = 1000
    
    count_rec = 0
    count_freq = 0
    count_burst = 0
    
    for i in range(TRACE_CONF['RECENCY']):
        op = create_recency_item(rng, count, LATENCY)
        arr.extend(op)
        count += 1
        count_rec += len(op)
    
    for i in range(TRACE_CONF['FREQUENCY']):
        op = create_frequency_item(rng, count, LATENCY)
        arr.extend(op)
        count += 1
        count_freq += len(op)
        
    for i in range(TRACE_CONF['BURSTINESS']):
        op = create_burstiness_item(rng, count, LATENCY)
        arr.extend(op)
        count += 1
        count_burst += len(op)
        
    for i in range(TRACE_CONF['ONE_HIT_WONDERS']):
        op = (random.randint(0, TOTAL_TICKS * TICK_TIME), count, 0, LATENCY)
        arr.append(op)
        count += 1
        
    print(f'Recency requests: {count_rec:,}\t|\tFrequency requests: {count_freq:,}\t|\tBurst requests: {count_burst:,}')
    
    arr = sorted(arr, key = lambda x : x[0])
    
    with open(f'synthetic.trace', 'w') as output_file:
        for row in arr:
            output_file.write(f'{row[0]} {row[1]} {row[2]} {row[3]}\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen-new-trace', help="Generate new synthetic trace", action='store_true', required=False)
    parser.add_argument('--gen-example', help="Generate the traffic example", action='store_true', required=False)
    parser.add_argument('--seed', help="Random seed for the generation", type=int, required=False)
    
    args = parser.parse_args()
    
    seed = args.seed if args.seed is not None else random.randint(1_000_000, 9_999_999) # 1872585
    print(f'seed: {seed:,}')
    rng : np.random.Generator = np.random.default_rng(seed=args.seed)
    random.seed(seed)
    
    if args.gen_new_trace:
        gen_trace(rng)
    
    if args.gen_example:
        plot_example(rng)
    

if __name__ == "__main__":
    main()