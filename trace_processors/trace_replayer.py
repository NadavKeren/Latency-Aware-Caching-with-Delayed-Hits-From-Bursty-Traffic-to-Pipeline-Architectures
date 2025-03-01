import argparse

from rich import pretty
from rich.console import Console
from rich.progress import Progress

from pathlib import Path
from os import SEEK_END, SEEK_CUR

from typing import List, Dict, TextIO

from itertools import chain, islice

from utils import *

pretty.install()
CONSOLE = Console()


CURR_EMPTY_ID = 0
TIMEFRAME = 0


def read_last_line(file_path: Path):
    path = Path(file_path)
    with path.open('rb') as file:
        file.seek(0, SEEK_END)
        pos = file.tell() - 2
        while pos >= 0:
            file.seek(pos)
            if file.read(1) == b'\n':
                break
            pos -= 1
        last_line = file.readline().decode()
    return last_line


def calculate_timeframe(file_path: Path) -> None:
    global TIMEFRAME
    with file_path.open('r') as file:
        first_line = file.readline()
    
    last_line = read_last_line(fname) # more efficient for larger files
    
    start_time = first_line.split(' ')[0]
    end_time = last_line.split(' ')[0]
    
    TIMEFRAME = int(end_time) - int(start_time)


def read_and_write_into_output(input_file: TextIO, output_file: TextIO, num_of_timeframes: int, should_change_keys: bool):
    global CURR_EMPTY_ID
    ids = dict()
    
    BATCH_SIZE = 10000
    lines = [line for line in islice(input_file, 0, BATCH_SIZE)]
    
    while(lines):
        for line in lines:
            splitted_line = line.split(' ')
            curr_timestamp = int(splitted_line[0]) + TIMEFRAME * num_of_timeframes + 1
            key = splitted_line[1]
            hit_penalty = splitted_line[2]
            delay = splitted_line[3]
            
            if (not should_change_keys): 
                new_id = key
            else :
                new_id = ids.get(key)
                
                if new_id is None:
                    new_id = CURR_EMPTY_ID
                    CURR_EMPTY_ID += 1
                    ids[key] = new_id
            
            output_file.write(f'{curr_timestamp} {new_id} {hit_penalty} {delay}')
    
        lines = [line for line in islice(input_file, 0, BATCH_SIZE)]
    
    ids.clear()


def pad_file(dir : str, file_path : Path, num_of_times : int, should_change_keys: bool):
    base_fname = file_path.name.rstrip('.trace')
    output_file_path =  Path(dir) / f'{base_fname}x{num_of_times + 1}.trace'
    
    calculate_timeframe(f'{dir}/{fname}')
    
    with Progress as progress, file_path.open('r') as origin_file, output_file_path.open('w') as output_file:
        padding = progress.add_task('[bold #bedcfe]Adding replays', total=num_of_times, start=True)
        for i in range(num_of_times + 1):
            read_and_write_into_output(origin_file, output_file, i, should_change_keys)
            origin_file.seek(0)
        
        progress.advance(padding, advance=1)
                    
        CONSOLE.print(f'[bold yellow]New number of unqiue items is {CURR_EMPTY_ID}, timeframe: {TIMEFRAME}')


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-d', '--directory', help='The directory of the input and expected output files', type=str, required=True)
    parser.add_argument('-f', '--file', help='The file to be padded, does not change the original', type=str, required=True)
    parser.add_argument('-t', '--times', help='Number of times to pad the file (1 -> doubles the file, 2 -> triples...)', type=int, required=True)
    parser.add_argument('--dont-change-keys', help='Passing this flag will use the original item keys', action='store_true', required=False)
    
    args = parser.parse_args()
    print(f'Given args: {args}')
    
    pad_file(args.directory, args.file, args.times, args.dont_change_keys is None)
    
    
if __name__ == '__main__':
    main()