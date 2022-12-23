#!/usr/bin/env python3

import subprocess

ep_process: list[int] = [4, 8, 16, 32, 64, 128, 256, 512]
ep_size_str: list[str] = ["S", "W", "A", "B", "C", "D", "E", "F"]
ep_size_int: list[int] = [24, 25, 28, 30, 32, 36, 40, 44]

list_csvDir: list[str] = ["csv_files/ep_1st/", "csv_files/ep_2nd/", "csv_files/ep_3rd/"]

for elem_csvDir in list_csvDir:
    for elem_process in ep_process:
        for elem_size in ep_size_str:
            fileName_before = f"{elem_csvDir}/ep_size{elem_size}_process{elem_process}.csv"
            fileName_after = f"{elem_csvDir}/ep_size{ep_size_int[ep_size_str.index(elem_size)]}_process{elem_process}.csv"
            list_command = ["cp", fileName_before ,fileName_after]
            subprocess.run(list_command, cwd="/root/src/")
