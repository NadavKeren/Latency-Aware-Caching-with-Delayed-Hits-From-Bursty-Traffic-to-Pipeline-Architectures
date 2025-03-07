from traces import *
from policies import *
import fire
import urllib
import os
import csv
import json
from subprocess import call
import subprocess
from pyhocon import ConfigFactory
from pyhocon import ConfigTree
from pyhocon import HOCONConverter
from enum import Enum
from pprint import pprint
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from typing import Union, List


with open(os.path.join(os.path.dirname(__file__), 'conf.json')) as conf_file:
    local_conf = json.load(conf_file)
caffeine_root = local_conf['caffeine_root']
resources_path = local_conf['resources'] if local_conf['resources'] else caffeine_root + 'simulator{0}src{0}main{0}resources{0}com{0}github{0}benmanes{0}caffeine{0}cache{0}simulator{0}parser'.format(os.sep)
output_path = local_conf['output'] if local_conf['output'] else os.getcwd() + os.sep
output_csvs_path = output_path + 'csvs' + os.sep


class Admission(Enum):
    ALWAYS = 'Always'
    TINY_LFU = 'TinyLfu'


def single_run(policy, trace:str=None, trace_files:Union[str, List[str]]=None, trace_folder:str=None, 
               trace_format:str=None, size:int=4, additional_settings:dict={}, name:str=None, 
               save:bool=True, reuse:bool=False, verbose:bool=False, readonly:bool=False, 
               seed:int=1033096058):
    if (trace is None and (trace_files is None or trace_folder is None or trace_format is None)):
        raise ValueError('Either trace or ALL trace_file, trace_folder and trace_format must be provided')
    
    name = name if name else f'{trace_files}-{size}-{policy}'
    policy = Policy[policy]
    
    if trace is not None:
        trace = Trace[trace]
        if 0 < size < 9:
            size = trace.typical_caches()[size-1]
    
    conf_path = caffeine_root + 'simulator{0}src{0}main{0}resources{0}'.format(os.sep)
    conf_file = conf_path + 'application.conf'
    
    if not os.path.exists(output_csvs_path):
        os.makedirs(output_csvs_path)
        
    run_simulator_cmd = './gradlew simulator:run -x caffeine:compileJava -x caffeine:compileCodeGenJava -PjvmArgs=-Xmx8g'
#   run_simulator_cmd = './gradlew simulator:run' 

    if os.path.exists(conf_file):
        conf = ConfigFactory.parse_file(conf_file)
    else:
        conf = ConfigFactory.parse_string("""
                                          caffeine {
                                            simulator {
                                            }
                                          }
                                          """)
    simulator = conf['caffeine']['simulator']
    
    if (trace is not None):
        if (trace_files is not None or trace_format is not None):
            print('Warning: Due to multiple definitions of trace origin, using only the default trace values')
            
        trace_files = trace.value['file']
        trace_folder = trace.value['format']
        trace_format = trace.value['format']
    
    if (type(trace_files) is list):
        paths = [resources_path + os.sep + trace_folder + os.sep + path for path in trace_files]
        simulator.put('files.paths', paths)
    else:
        simulator.put('files.paths', [ resources_path + os.sep + trace_folder + os.sep + trace_files ])
    
    simulator.put('files.format', trace_format)
    simulator.put('maximum-size', size)
    simulator.put('policies', [ policy.value ])
    simulator.put('admission', [ Admission.ALWAYS.value ])
    simulator.put('random-seed', seed)
    
    if verbose:
        simulator.put('report.format', 'table')
        simulator.put('report.output', 'console')
    else:
        simulator.put('report.format', 'csv')
        simulator.put('report.output', output_csvs_path + f'{name}.csv')

    for k,v in additional_settings.items():
        simulator.put(k,v)

    with open(conf_file, 'w') as f:
        f.write(HOCONConverter.to_hocon(conf))
    if (not reuse or not os.path.isfile(simulator['report']['output'])) and not readonly:
        retcode = call(run_simulator_cmd, shell = True, cwd = caffeine_root, stdout = subprocess.DEVNULL if not verbose else None)
        if (not retcode == 0):
            return False
    
    if not verbose:
        with open(simulator['report']['output'], 'r') as csvfile:
            results = pd.read_csv(csvfile)
        
        if not save:
            os.remove(simulator['report']['output'])
            
        return results


def download_single_trace(trace, path=None):
        if not path:
            path = resources_path
        if path[-1] != os.sep:
            path += os.sep
        if not os.path.exists(path + trace.format()):
            os.makedirs(path + trace.format())
        print('Downloading ' + trace.name + '...')
        urllib.request.urlretrieve(trace.url() + '?dl=1', path + trace.format() + os.sep + trace.file())
        print(trace.name + ' downloaded')

def parse_traces(traces):
    if not traces:
        return [trace for trace in Trace]
    if type(traces) == str:
        traces = [ traces ]
    if type(traces) == list:
        return [ Trace[trace] for trace in traces ]
    return []

def rf_rank(trace, size):
    lru = single_run('lru', trace.name, size, save=True, reuse=True)
    lfu = single_run('lfu', trace.name, size, save=True, reuse=True)
    opt = single_run('opt', trace.name, size, save=True, reuse=True)
    return (lru-lfu)/opt 

class Tools(object):

    def download_traces(self, traces=None, path=None):
        """ 
        Download all traces to the given path, if the path is empty - download to caffeine resources path.
        Each trace placed into a subdirectory with the format name.
        """
        traces = parse_traces(traces)
        for trace in traces:
            download_single_trace(trace, path)

    def list_traces(self, sizes=False):
        """
        Print all the avaliable traces.
        """
        print('Avaliable traces are:')
        line = ' ' + '-'*(66 + 16 * (2 if sizes else 1) - 1)
        text ='|{:^15}|{:^65}|' + ('' if not sizes else '{:^15}|') 
        headers = ['Trace Name','Typical Cache Sizes'] + ([] if not sizes else ['Size']) 
        print(line)
        print(text.format(*headers))
        print(line)
        for trace in Trace:
            texts = [trace.name, str(trace.typical_caches())[1:-1]]
            if sizes:
                try:
                    with open(output_csvs_path + '{}-{}-{}.csv'.format(trace.name,trace.typical_caches()[0], 'lru'), 'r') as csvfile:
                        reader = csv.DictReader(csvfile)
                        result = int(next(reader)['Requests'])
                        texts.append('{:,}'.format(result))
                except:
                    texts.append('no lru csv')
            print(text.format(*texts))
        print(line)

    def run(self, policy, trace, size=4, changes={}, name=None, save=True, reuse=False, verbose=False):
        res = single_run(policy, trace, size, changes, name, save, reuse, verbose)
        print('The hit rate of {} on {} with cache size of {} is: {}%'
                .format(name if name else policy, trace, size if size > 8 else Trace[trace].typical_caches()[size-1], res))

    def battle(self, policy1, policy2, changes1={}, changes2={}, name1=None, name2=None, save=True, reuse=False, verbose=False, rfo=False, filt=None, losers=False):
        self.compare(policies=[policy1, policy2], changes=[changes1, changes2], names=[name1, name2], save=save, reuse=reuse, verbose=verbose, rfo=rfo, filt=filt, losers=losers)

    def compare(self, policies, changes=None, names=None, save=True, reuse=False, verbose=False, rfo=False, filt=None, losers=False):
        if not changes or not changes[0]:
            changes = [{}]*len(policies)
        if not names or not names[0]:
            names = policies
        policies_wins = [0]*len(policies)

        columns = 3 + len(policies) + (1 if rfo else 0)
        line = ' ' + '-'*(16 * columns - 1)
        text ='|' + '{:^15}|'*columns 
        print(line)
        headers = ['Trace','Cache Size'] + (['(LRU-LFU)/OPT'] if rfo else []) 
        print(text.format(*headers, *names, 'Difference'))
        print(line)

        for trace in Trace:
            for size in range(1,1+8):
                policies_hr = [ single_run(policy, trace.name, size, change, name, save, reuse, verbose) \
                                for policy, change, name in zip(policies, changes, names)]
                texts = [trace.name, trace.typical_caches()[size-1]] + (['{:2.2f}'.format(rf_rank(trace, size))] if rfo else []) + \
                        ['{:2.2f}%'.format(policy_hr) for policy_hr in policies_hr] + ['{:2.2f}%'.format(max(policies_hr)-min(policies_hr))]

                if (filt and (abs(min(policies_hr) - max(policies_hr)) < filt)):
                    continue

                min_index = policies_hr.index(min(policies_hr))
                max_index = policies_hr.index(max(policies_hr))
                
                offset = 2 + (1 if rfo else 0)
                if not losers:
                    texts[offset + min_index] = ''
                texts[offset + max_index] = '\N{CHECK MARK} ' + texts[offset + max_index] 

                if min_index == max_index and not losers:
                    for i in range(len(policies_hr)):
                        texts[offset + i] = ''

                print(text.format(*texts))

                if min_index != max_index:
                    policies_wins[max_index] += 1

        print(line)
        print(text.format(*(['']*(1 + (1 if rfo else 0))),'Total Wins:', *policies_wins, ''))
        print(line)

if __name__ == '__main__':
    fire.Fire(Tools)
